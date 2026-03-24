from logging import config
from typing import Tuple, Dict, Any
import os
from networkx import config
from datetime import datetime
import torch
from dotenv import load_dotenv
from torch import optim, split, split
from huggingface_hub import HfApi
import sys

from torch.utils.data import DataLoader
from src.data_loading.HFDataloader import initialize_streaming_dataloader, initialize_dataloaders
from src.data_postprocessing.model_testing import model_testing
from src.model_training.metric_functions import all_statistics
from src.model_training.utils import _train_single_batch, get_device, get_subset, training_loop
from src.utils.training_config import TrainingConfig, TestConfig
from src.utils.logger import setup_logger, replace_output
import datasets
import torch.nn as nn
import os


def setup_directories_and_logging(config: TrainingConfig, model_name: str) -> Tuple[str, object]:
    """Create model directory and setup logging"""
    time_string = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.getenv("OUTPUT_DIR")
    model_dir = os.path.join(output_dir, "models", f"{model_name}", time_string)
    os.makedirs(model_dir, exist_ok=True)

    logger = setup_logger(
        log_file="outputlog.txt",
        log_dir=model_dir,
        logging_enabled=True,
        console_enabled=False
    )
    replace_output(logger)

    return model_dir, logger


def save_results(model_state_dict, model_dir: str, model_name: str, metrics):
    """Save model, metrics, and test accuracy"""
    filepath = os.path.join(model_dir, f"{model_name}.pth")
    torch.save(model_state_dict, filepath)
    print(f"Model saved to: {filepath}")

    plot_path = filepath.replace('.pth', '.png')
    metric_path = filepath.replace('.pth', '.metrics')
    metrics.save_plot("Model Performance", plot_path)
    metrics.save_metrics(metric_path)

    print(f"Plot saved to: {plot_path}")

def _warmup(model, dataloader, optimizer, criterion, device, steps=5):
    train_iter = iter(dataloader)
    for _ in range(steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            break
        _train_single_batch(model, batch, optimizer, criterion, device)
    torch.cuda.synchronize()

def train_model(config: TrainingConfig, model_name: str):
    """Main training function - single entry point for all models"""
    load_dotenv() # Load environment variables from .env file

    # Setup
    run_dir, logger = setup_directories_and_logging(config, model_name)
    print(f"Training {config.model_class.__name__}. Model will be saved to: {run_dir}")
    print(f"Weight Decay: {config.weight_decay}, Learning Rate: {config.learning_rate}")
    print(f"Max Num Epochs: {config.max_nr_epochs}")
    if config.max_train_steps is not None:
        print(f"Max Train Steps: {config.max_train_steps}")
    config.save(os.path.join(run_dir))

    # Initialize model
    model = config.model_class(**config.model_args)

        
    # Load and prepare data
    dataloaders = initialize_dataloaders(config)

    # Setup training
    device = get_device()
    criterion = config.criterion()
    optimizer = config.optimizer(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    model.to(device)
    if config.compile_model:
        print("Compiling model with torch.compile() for potentially faster training.")
        model = torch.compile(model)
        print("Warming up compiled model...")
        _warmup(model, dataloaders["train"], optimizer, criterion, device, steps=5)
    
    

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
        output_dir=run_dir,
        max_batches=config.test_max_batches,
    )
    save_results(model_state_dict, run_dir, model_name, metrics)
    # cleanup
    del model
    torch.cuda.empty_cache()
    print("Training complete.")



