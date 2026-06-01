import os
import time
import threading
from contextlib import nullcontext
import datasets

import torch
from src.model_training.Metrics import Metrics
from src.model_training.batch_preprocessing import prepare_model_batch
from src.model_training.metric_functions import calculate_accuracy_and_loss
from sklearn.metrics import confusion_matrix

from src.utils.logger import get_logger
logging = get_logger() 
def save_model(model, model_folder, model_name):
    os.makedirs(model_folder, exist_ok=True)
    model_path = os.path.join(model_folder, f"{model_name}.pth")
    torch.save(model, model_path)
    logging.info(f"Saved full model to {model_path}")
    return model_path


def load_model(model_filepath, device=None):
    if device is None:
        device = torch.device("cpu")
    model = torch.load(model_filepath, map_location=device, weights_only=False)
    model.eval()
    return model

def model_name(model_name_prefix):
    # current date and time
    date = time.strftime("%Y%m%d-%H%M%S")
    model_name = f"{model_name_prefix}_{date}"
    return model_name
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("WARNING: Training on CPU, this may be slow. Consider using a GPU for faster training.")
    return device

def get_subset(dataset, fraction, shuffle=True, seed=42):
    if 0.9999 < fraction <= 1.0:
        return dataset
    if fraction <= 0 or fraction > 1.0001:
        raise ValueError("Fraction must be between 0 and 1.")
    dataset_size = int(max(len(dataset) * fraction, 1))
    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    return dataset.select(range(dataset_size))
def clear_cache(dataset_dir):
    ds = datasets.load_from_disk(dataset_dir)
    ds.cleanup_cache_files()
    print(f"Cleared cache files in dataset at {dataset_dir}")



def train_val_test_split(dataset, train_fraction=0.7, val_fraction=0.15, seed=42):
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("Train and validation fractions must sum to less than 1.0")

    # First split into train and temp (val + test)
    temp_fraction = 1.0 - train_fraction
    train_val_test = dataset.train_test_split(test_size=temp_fraction, seed=seed, shuffle=True)
    train_set = train_val_test['train']
    temp_set = train_val_test['test']

    # Now split temp into validation and test
    test_fraction = 1.0 - (val_fraction / temp_fraction)
    val_test = temp_set.train_test_split(test_size=test_fraction, seed=seed, shuffle=True)
    val_set = val_test['train']
    test_set = val_test['test']

    return {"train": train_set, "val": val_set,"test": test_set}

def training_phase(model, trainloader, optimizer, criterion, device):
    """Train over a finite dataloader and return aggregate accuracy/loss."""
    model.train()
    correct = 0
    total = 0
    running_loss = 0.0
    num_batches = 0
    for i, batch in enumerate(trainloader, 0):
        inputs, labels = prepare_model_batch(batch, device, scramble=True)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        running_loss += loss.item()
        num_batches += 1

    train_loss = running_loss / max(num_batches, 1)
    train_acc = correct / max(total, 1)
    return train_acc, train_loss


def _train_single_batch(model, batch, optimizer, criterion, device):
    """Train one batch and return correct predictions, sample count and loss."""
    model.train()
    inputs, labels = prepare_model_batch(batch, device, scramble=True)

    return _train_single_batch_prepared(model, inputs, labels, optimizer, criterion)


def _train_single_batch_prepared(model, inputs, labels, optimizer, criterion):
    """Train one already-prepared batch and return correct predictions, sample count and loss."""
    model.train()

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    _, predicted = torch.max(outputs, 1)
    batch_total = labels.size(0)
    batch_correct = (predicted == labels).sum().item()
    return batch_correct, batch_total, loss.item()


def _resolve_amp_dtype(amp_dtype):
    if isinstance(amp_dtype, torch.dtype):
        return amp_dtype
    if isinstance(amp_dtype, str):
        normalized = amp_dtype.strip().lower()
        if normalized in {"fp16", "float16", "half"}:
            return torch.float16
        if normalized in {"bf16", "bfloat16"}:
            return torch.bfloat16
    return torch.bfloat16


def _train_single_batch_prepared_amp(
    model,
    inputs,
    labels,
    optimizer,
    criterion,
    device,
    use_mixed_precision=False,
    amp_dtype=torch.bfloat16,
    grad_scaler=None,
):
    """Train one already-prepared batch with optional CUDA AMP and return stats."""
    model.train()
    optimizer.zero_grad()

    use_amp = bool(use_mixed_precision and device.type == "cuda")
    amp_context = (
        torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)
        if use_amp
        else nullcontext()
    )

    with amp_context:
        outputs = model(inputs)
        loss = criterion(outputs, labels)

    if not torch.isfinite(loss):
        optimizer.zero_grad(set_to_none=True)
        return 0, 0, float("nan")

    if use_amp and grad_scaler is not None:
        grad_scaler.scale(loss).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()
    else:
        loss.backward()
        optimizer.step()

    _, predicted = torch.max(outputs, 1)
    batch_total = labels.size(0)
    batch_correct = (predicted == labels).sum().item()
    return batch_correct, batch_total, loss.item()


class CUDABatchPrefetcher:
    """Prefetch and preprocess the next batch on a dedicated CUDA stream."""

    def __init__(self, dataloader, device, scramble=True, batch_preparation_fn=prepare_model_batch):
        self.dataloader = dataloader
        self.device = device
        self.scramble = scramble
        self.batch_preparation_fn = batch_preparation_fn
        self.stream = torch.cuda.Stream(device=device)
        self.loader_iter = None
        self.next_inputs = None
        self.next_labels = None
        self._prefetch_thread = None
        self._thread_error = None

    def reset(self):
        self.loader_iter = iter(self.dataloader)
        self._thread_error = None
        self._prefetch_async()

    def _prefetch_worker(self):
        try:
            batch = next(self.loader_iter)
        except StopIteration:
            self.next_inputs = None
            self.next_labels = None
            return
        except Exception as exc:
            self._thread_error = exc
            self.next_inputs = None
            self.next_labels = None
            return

        with torch.cuda.stream(self.stream):
            inputs, labels = self.batch_preparation_fn(batch, self.device, scramble=self.scramble)
            self.next_inputs = inputs
            self.next_labels = labels

    def _prefetch_async(self):
        self._prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self._prefetch_thread.start()

    def _wait_prefetch(self):
        if self._prefetch_thread is not None:
            self._prefetch_thread.join()
            self._prefetch_thread = None

        if self._thread_error is not None:
            err = self._thread_error
            self._thread_error = None
            raise err

    def next(self):
        self._wait_prefetch()
        if self.next_inputs is None:
            raise StopIteration

        current_stream = torch.cuda.current_stream(self.device)
        current_stream.wait_stream(self.stream)

        inputs = self.next_inputs
        labels = self.next_labels

        if torch.is_tensor(inputs) and inputs.is_cuda:
            inputs.record_stream(current_stream)
        if torch.is_tensor(labels) and labels.is_cuda:
            labels.record_stream(current_stream)

        # Start preparing the following batch while current batch is being trained.
        self._prefetch_async()
        return inputs, labels


def _safe_len(dataloader):
    """Return dataloader length when available, otherwise None."""
    try:
        return len(dataloader)
    except (TypeError, AttributeError):
        return None


def training_loop(
    model,
    trainloader,
    validationloader,
    optimizer,
    criterion,
    model_folder,
    scheduler=None,
    max_epochs=200,
    max_train_steps=None,
    steps_per_epoch=1000,
    eval_every_steps=1000,
    log_every_steps=100,
    use_cuda_prefetcher=True,
    batch_preparation_fn=prepare_model_batch,
    use_mixed_precision=False,
    amp_dtype="bfloat16",
    validation_max_batches=None,
    patience=10,
    minimum_delta=0.001,
    time_limit=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved_amp_dtype = _resolve_amp_dtype(amp_dtype)
    use_amp = bool(use_mixed_precision and device.type == "cuda")
    use_grad_scaler = bool(use_amp and resolved_amp_dtype == torch.float16)
    grad_scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)

    metric_path =os.path.join(model_folder, f'metrics_training_loop.pt')
    plot_path = os.path.join(model_folder, f'plots_training_loop.png')
    logging.info(f"Using device: {device}")
    logging.info(f"Using metrics path: {metric_path}")
    logging.info(f"Using patience: {patience}")

    trainloader_len = _safe_len(trainloader)
    validationloader_len = _safe_len(validationloader)
    if trainloader_len is not None:
        logging.info(f"Training batches per pass: {trainloader_len}")
    else:
        logging.info("Training loader is streaming/iterable (unknown length per pass).")

    if validationloader_len is not None:
        logging.info(f"Validation batches per pass: {validationloader_len}")
    else:
        logging.info("Validation loader length is unknown.")


    if max_train_steps is None:
        if trainloader_len is not None and max_epochs is not None:
            max_train_steps = max_epochs * trainloader_len
            steps_per_epoch = trainloader_len
            eval_every_steps = trainloader_len
        else:
            max_train_steps = max_epochs * steps_per_epoch

    eval_every_steps = max(1, eval_every_steps)
    log_every_steps = max(1, log_every_steps)
    steps_per_epoch = max(1, steps_per_epoch)

    logging.info(f"Using max_train_steps: {max_train_steps}")
    logging.info(f"Using steps_per_epoch: {steps_per_epoch}")
    logging.info(f"Using eval_every_steps: {eval_every_steps}")
    logging.info(f"Using log_every_steps: {log_every_steps}")

    # time limit in readable format for logging
    if time_limit:
        time_limit_str = f"{time_limit // 3600}h {(time_limit % 3600) // 60}m {time_limit % 60}s"
        logging.info(f"Using time limit: {time_limit_str}.")
    logging.info(f"Using plot path: {plot_path}")

    if device == torch.device("cpu"):
        logging.warning("Training on CPU, this may be slow. Consider using a GPU for faster training.")


    best_model_state_dict = None
    epochs_best_model = None
    metrics = Metrics(patience=patience, minimum_delta=minimum_delta)

    start_time = time.time()

    global_step = 0
    train_iter = iter(trainloader)
    prefetcher = None
    if device.type == "cuda" and use_cuda_prefetcher:
        logging.info("CUDA prefetcher enabled: overlapping next-batch preprocessing with current compute.")
        prefetcher = CUDABatchPrefetcher(
            trainloader,
            device=device,
            scramble=True,
            batch_preparation_fn=batch_preparation_fn,
        )
        prefetcher.reset()
    interval_correct = 0
    interval_total = 0
    interval_loss = 0.0
    interval_batches = 0
    interval_data_wait = 0.0
    interval_prep_time = 0.0
    interval_compute_time = 0.0
    interval_start = time.time()
    logging.info("Starting training loop...")
    current_best_model_path = None
    amp_fallback_applied = False
    while global_step < max_train_steps:
        model.train()
        data_wait_start = time.time()
        try:
            if prefetcher is not None:
                inputs, labels = prefetcher.next()
            else:
                # Get next batch, or restart iterator if we've reached end of dataloader.
                batch = next(train_iter)
        except StopIteration:
            if prefetcher is not None:
                prefetcher.reset()
                inputs, labels = prefetcher.next()
            else:
                train_iter = iter(trainloader)
                batch = next(train_iter)
        except RuntimeError as exc:
            if "DataLoader worker" in str(exc):
                raise RuntimeError(
                    "Training dataloader worker crashed. This is often caused by too many workers "
                    "or worker-side exceptions in transforms. Try lowering TrainingConfig.num_cpus "
                    "(for streamed datasets start with 0-2 workers)."
                ) from exc
            raise

        interval_data_wait += time.time() - data_wait_start

        prep_start = time.time()
        if prefetcher is not None:
            # In prefetch mode, prep reflects stream sync + handoff cost.
            interval_prep_time += time.time() - prep_start
            compute_start = time.time()
            batch_correct, batch_total, batch_loss = _train_single_batch_prepared_amp(
                model,
                inputs,
                labels,
                optimizer,
                criterion,
                device,
                use_mixed_precision=use_amp,
                amp_dtype=resolved_amp_dtype,
                grad_scaler=grad_scaler,
            )
        else:
            prepared_inputs, prepared_labels = batch_preparation_fn(batch, device, scramble=True)
            interval_prep_time += time.time() - prep_start

            compute_start = time.time()
            batch_correct, batch_total, batch_loss = _train_single_batch_prepared_amp(
                model,
                prepared_inputs,
                prepared_labels,
                optimizer,
                criterion,
                device,
                use_mixed_precision=use_amp,
                amp_dtype=resolved_amp_dtype,
                grad_scaler=grad_scaler,
            )
            prepared_inputs = None
            prepared_labels = None

        if not torch.isfinite(torch.tensor(batch_loss)):
            if use_amp and not amp_fallback_applied:
                logging.warning(
                    "Non-finite loss detected with AMP; disabling mixed precision for the rest of this run."
                )
                use_amp = False
                grad_scaler = torch.amp.GradScaler("cuda", enabled=False)
                amp_fallback_applied = True
                continue

            logging.warning("Skipping non-finite batch loss.")
            continue

        interval_compute_time += time.time() - compute_start
        global_step += 1

        interval_correct += batch_correct
        interval_total += batch_total
        interval_loss += batch_loss
        interval_batches += 1

        if global_step % log_every_steps == 0:
            running_acc = interval_correct / max(interval_total, 1)
            running_loss = interval_loss / max(interval_batches, 1)
            avg_data_wait_ms = (interval_data_wait / max(interval_batches, 1)) * 1000
            avg_prep_ms = (interval_prep_time / max(interval_batches, 1)) * 1000
            avg_compute_ms = (interval_compute_time / max(interval_batches, 1)) * 1000
            total_interval_time = max(interval_data_wait + interval_prep_time + interval_compute_time, 1e-9)
            samples_per_sec = interval_total / total_interval_time
            data_wait_fraction = interval_data_wait / total_interval_time
            prep_fraction = interval_prep_time / total_interval_time
            compute_fraction = interval_compute_time / total_interval_time
            logging.info(
                f"Step {global_step}/{max_train_steps} | "
                f"Train Acc: {running_acc * 100:.4f}% | "
                f"Train Loss: {running_loss:.4f} | "
                f"Avg Data Wait: {avg_data_wait_ms:.1f}ms | "
                f"Avg Prep: {avg_prep_ms:.1f}ms | "
                f"Avg Compute: {avg_compute_ms:.1f}ms | "
                f"Samples/s: {samples_per_sec:.1f} | "
                f"Shares(wait/prep/compute): "
                f"{data_wait_fraction * 100:.1f}%/"
                f"{prep_fraction * 100:.1f}%/"
                f"{compute_fraction * 100:.1f}%"
            )
        if prefetcher is not None:
            inputs = None
            labels = None
        else:
            batch = None  # Free batch memory
        should_eval = (global_step % eval_every_steps == 0) or (global_step == max_train_steps)
        if should_eval:
            model.eval()
            

            train_acc = interval_correct / max(interval_total, 1)
            train_loss = interval_loss / max(interval_batches, 1)
            try:
                val_acc, val_loss = calculate_accuracy_and_loss(
                    model,
                    validationloader,
                    criterion,
                    device,
                    max_batches=validation_max_batches,
                    batch_preparation_fn=batch_preparation_fn,
                    use_mixed_precision=use_amp,
                    amp_dtype=resolved_amp_dtype,
                )
            except RuntimeError as exc:
                if "DataLoader worker" in str(exc):
                    raise RuntimeError(
                        "Validation dataloader worker crashed at evaluation. "
                        "Reduce TrainingConfig.num_cpus and/or set validation workers to 0 for streamed data."
                    ) from exc
                raise
            if scheduler:
                scheduler.step()

            metrics.update(train_acc * 100, train_loss, val_acc * 100, val_loss)
            metrics.save_metrics(metric_path)
            metrics.save_plot("Training and Validation Metrics", plot_path)

            pseudo_epoch = (global_step - 1) // steps_per_epoch + 1
            average_time_per_eval = (time.time() - start_time) / len(metrics.training_accuracy)
            interval_time = time.time() - interval_start

            logging.info(f'===============Eval checkpoint at step {global_step} (pseudo-epoch {pseudo_epoch})===============')
            logging.info(f'\tTraining  \tAccuracy: {metrics.training_accuracy[-1]:.4f}%\tLoss: {metrics.train_loss[-1]:.4f}')
            logging.info(f'\tValidation\tAccuracy: {metrics.validation_accuracy[-1]:.4f}%\tLoss: {metrics.validation_loss[-1]:.4f}')
            logging.info('-' * 100)

            improved = metrics.model_improved()
            if improved:
                # remove old best model file if it exists
                if current_best_model_path and os.path.exists(current_best_model_path):
                    os.remove(current_best_model_path)
                    logging.info(f"\tRemoved old best model at {current_best_model_path}")
                current_best_model_path = save_model(model, model_folder, f'current_best_model_epoch_{pseudo_epoch}.pth')
                logging.info(f"\tSaved best model at step {global_step} to {current_best_model_path}")
                epochs_best_model = pseudo_epoch
            else:
                metrics.patience_counter += 1

            logging.info(f'\tPatience Counter: {metrics.patience_counter}/{patience}')
            logging.info(
                f'\tTime for eval interval: {interval_time // 60:.2f}m {interval_time % 60:.0f}s\t'
                f'(avg: {average_time_per_eval // 60}m {average_time_per_eval % 60:.0f}s/check)\t'
                f'Time elapsed since start: {(time.time() - start_time) // 60:.0f}m'
            )
            logging.info(
                f'\tPipeline profile: data_wait={interval_data_wait:.2f}s, '
                f'prep={interval_prep_time:.2f}s, '
                f'compute={interval_compute_time:.2f}s, '
                f'data_wait_share={100 * interval_data_wait / max(interval_data_wait + interval_prep_time + interval_compute_time, 1e-9):.1f}%'
            )
            if scheduler:
                logging.info(f'\tLearning Rate: {optimizer.param_groups[0]["lr"]:.2e}')
            logging.info('=' * 100)

            interval_correct = 0
            interval_total = 0
            interval_loss = 0.0
            interval_batches = 0
            interval_data_wait = 0.0
            interval_prep_time = 0.0
            interval_compute_time = 0.0
            interval_start = time.time()

            if metrics.patience_counter >= patience:
                logging.info("Early stopping due to overfitting.")
                break

        if time_limit and (time.time() - start_time) > time_limit:
            logging.info("Time limit reached, stopping training.")
            break

    if best_model_state_dict is None:
        best_model_state_dict = model.state_dict()
        epochs_best_model = (global_step - 1) // steps_per_epoch + 1 if global_step > 0 else 0

    logging.info('Finished Training')
    return best_model_state_dict, epochs_best_model,metrics



