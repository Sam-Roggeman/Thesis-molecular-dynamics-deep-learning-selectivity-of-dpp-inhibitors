from typing import Tuple, Dict, Any
import os
import gc
from datetime import datetime
import torch
from dotenv import load_dotenv
from torch import optim, split, split
from huggingface_hub import HfApi
import sys
from torch import optim, split
from torch.utils.data import DataLoader
from src.data_loading.HFDataloader import initialize_dataloaders
from src.data_postprocessing.model_testing import model_testing
from src.model_training.batch_preprocessing import prepare_model_batch
from src.model_training.metric_functions import all_statistics
from src.model_training.utils import _train_single_batch, get_device, get_subset, training_loop
from src.utils.cacheManager import cacheManager
from src.utils.training_config import TrainingConfig, TestConfig
import datasets
import torch.nn as nn
import os
from src.utils.logger import init_logger, get_logger, DEBUG as LOGGING_DEBUG
logging = get_logger() 

def _is_cuda_oom_error(exc: BaseException) -> bool:
    """Return True when exception indicates a CUDA OOM condition."""
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return "cuda out of memory" in msg or "cudnn_status_alloc_failed" in msg


def _cleanup_cuda_memory():
    """Release cached CUDA memory between retries."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass





def save_results(model_state_dict, model_dir: str, model_name: str, metrics):
    """Save model, metrics, and test accuracy"""
    filepath = os.path.join(model_dir, f"{model_name}.pth")
    torch.save(model_state_dict, filepath)
    logging.info(f"Model saved to: {filepath}")

    plot_path = filepath.replace('.pth', '.png')
    metric_path = filepath.replace('.pth', '.metrics')
    metrics.save_plot("Model Performance", plot_path)
    metrics.save_metrics(metric_path)

    logging.info(f"Plot saved to: {plot_path}")

def _warmup(model, dataloader, optimizer, criterion, device, steps=5):
    train_iter = iter(dataloader)
    for step_idx in range(steps):
        fetch_start = datetime.now()
        try:
            batch = next(train_iter)
        except StopIteration:
            break
        fetch_elapsed = (datetime.now() - fetch_start).total_seconds()
        step_start = datetime.now()
        model.train()
        inputs, labels = prepare_model_batch(batch, device, scramble=True)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        # Warmup is for graph capture/compilation only: do not update weights here.
        optimizer.zero_grad(set_to_none=True)
        step_elapsed = (datetime.now() - step_start).total_seconds()
        logging.info(f"Warmup step {step_idx + 1}/{steps}: fetch={fetch_elapsed:.2f}s, train_step={step_elapsed:.2f}s")
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _warmup_with_batch_fn(model, dataloader, optimizer, criterion, device, batch_preparation_fn, steps=5):
    train_iter = iter(dataloader)
    for step_idx in range(steps):
        fetch_start = datetime.now()
        try:
            batch = next(train_iter)
        except StopIteration:
            break
        fetch_elapsed = (datetime.now() - fetch_start).total_seconds()
        step_start = datetime.now()
        model.train()
        inputs, labels = batch_preparation_fn(batch, device, scramble=True)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.zero_grad(set_to_none=True)
        step_elapsed = (datetime.now() - step_start).total_seconds()
        logging.info(f"Warmup step {step_idx + 1}/{steps}: fetch={fetch_elapsed:.2f}s, train_step={step_elapsed:.2f}s")
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def initialize_run_directory(model_name):
    """Create model directory and setup logging"""
    time_string = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.getenv("OUTPUT_DIR")
    model_dir = os.path.join(output_dir, "models", f"{model_name}", time_string)
    os.makedirs(model_dir, exist_ok=True)
    return model_dir
    

def train_model(config: TrainingConfig, model_name: str, streaming: bool = False):
    """Main training function - single entry point for all models"""
    load_dotenv() # Load environment variables from .env file
    batch_preparation_fn = config.batch_preparation_fn or prepare_model_batch
    
    # Setup
    run_dir = initialize_run_directory(model_name)
    logger = init_logger(model_dir=run_dir, log_mode=LOGGING_DEBUG, log_file="debug.log")
    logger.info(f"Starting training for {model_name} with, saving to {run_dir}")
    logger.info(f"Training configuration: {config}")
    logger.info(f"Weight Decay: {config.weight_decay}, Learning Rate: {config.learning_rate}")
    logger.info(f"Max Num Epochs: {config.max_nr_epochs}")
    if config.max_train_steps is not None:
        logger.info(f"Max Train Steps: {config.max_train_steps}")
    cache_manager = cacheManager(os.getenv("HF_CACHE_DIR"), os.getenv("FAST_CACHE_DIR"))

    max_retries = max(0, int(config.oom_max_retries)) if config.oom_retry_enabled else 0
    attempt = 0
    try:
        while True:
            model = None
            dataloaders = None
            try:
                config.save(os.path.join(run_dir))

                # Initialize model and data for this attempt (batch size may change after OOM).
                model = config.model_class(**config.model_args)
                dataloaders = initialize_dataloaders(config, cache_manager, streaming=streaming)

                # Setup training
                device = get_device()
                criterion = config.criterion()
                if isinstance(criterion, nn.Module):
                    criterion = criterion.to(device)
                optimizer = config.optimizer(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
                effective_amp_dtype = config.amp_dtype

                # Set up device and CUDA settings before moving model to device
                if device.type == "cuda":
                    # Throughput-oriented CUDA backend settings.
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    torch.set_float32_matmul_precision("high")
                    if config.use_mixed_precision:
                        dtype_name = str(config.amp_dtype).strip().lower()
                        wants_bf16 = dtype_name in {"bf16", "bfloat16", "torch.bfloat16"}
                        if wants_bf16 and not torch.cuda.is_bf16_supported():
                            effective_amp_dtype = "float16"
                            logger.warning(
                                "Requested AMP dtype bfloat16, but CUDA device does not support native bf16. "
                                "Falling back to float16."
                            )

                model.to(device)
                if config.compile_model:
                    logger.info("Compiling model with torch.compile() for potentially faster training.")
                    model = torch.compile(model)
                    warmup_steps = max(0, int(getattr(config, "compile_warmup_steps", 1)))
                    if warmup_steps > 0:
                        logger.info(f"Warming up compiled model for {warmup_steps} step(s)...")
                        _warmup_with_batch_fn(
                            model,
                            dataloaders["train"],
                            optimizer,
                            criterion,
                            device,
                            batch_preparation_fn=batch_preparation_fn,
                            steps=warmup_steps,
                        )
                    else:
                        logger.info("Skipping explicit compile warmup (compile_warmup_steps=0).")

                # Train
                model_state_dict, nr_epochs, metrics = training_loop(
                    model=model,
                    model_folder=run_dir,
                    trainloader=dataloaders["train"],
                    validationloader=dataloaders["validation"],
                    optimizer=optimizer,
                    criterion=criterion,
                    max_epochs=config.max_nr_epochs,
                    max_train_steps=config.max_train_steps,
                    steps_per_epoch=config.steps_per_epoch,
                    eval_every_steps=config.eval_every_steps,
                    log_every_steps=config.log_every_steps,
                    use_cuda_prefetcher=config.use_cuda_prefetcher,
                    batch_preparation_fn=batch_preparation_fn,
                    use_mixed_precision=config.use_mixed_precision,
                    amp_dtype=effective_amp_dtype,
                    validation_max_batches=config.validation_max_batches,
                    patience=config.patience,
                    time_limit=config.time_limit
                )

                model.load_state_dict(model_state_dict)

                model_testing(
                    model,
                    dataloaders["test"],
                    criterion,
                    device,
                    batch_preparation_fn=batch_preparation_fn,
                    use_mixed_precision=config.use_mixed_precision,
                    amp_dtype=effective_amp_dtype,
                    output_dir=run_dir,
                    max_batches=config.test_max_batches,
                )
                save_results(model_state_dict, run_dir, model_name, metrics)
                logger.info("Training complete.")
                break

            except Exception as exc:
                is_oom = _is_cuda_oom_error(exc)
                can_retry = (
                    is_oom
                    and torch.cuda.is_available()
                    and config.oom_retry_enabled
                    and attempt < max_retries
                    and config.batch_size > max(1, int(config.min_batch_size))
                )

                if not can_retry:
                    raise

                old_batch_size = int(config.batch_size)
                new_batch_size = max(int(config.min_batch_size), old_batch_size // 2)

                if new_batch_size >= old_batch_size:
                    raise

                attempt += 1
                logger.info(
                    f"CUDA OOM detected (attempt {attempt}/{max_retries}). "
                    f"Reducing batch size from {old_batch_size} to {new_batch_size} and retrying..."
                )
                config.batch_size = new_batch_size
                _cleanup_cuda_memory()
            except SystemExit as e:
                # Catch and log unexpected SystemErrors that may occur during training (e.g., from torch.compile internals).
                logger.info(f"Job halted with exit code {e.code}")
                # Re-raise to allow external handlers (e.g., job schedulers) to detect the exit condition.
                raise
            finally:
                if model is not None:
                    del model
                if dataloaders is not None:
                    del dataloaders
                _cleanup_cuda_memory()
    except Exception as final_exc:
        logger.info(f"Training failed after {attempt} attempt(s) with batch size {config.batch_size}.")
        raise final_exc
    finally:
        cache_manager.cleanup()

