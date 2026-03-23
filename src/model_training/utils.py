import os
import time
import datasets

import torch

from src.model_training.Metrics import Metrics
from src.model_training.metric_functions import calculate_accuracy_and_loss
from sklearn.metrics import confusion_matrix
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
def load_model(model_class, model_filepath):
    model = model_class()
    model.load_state_dict(torch.load(model_filepath))
    return model
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
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = batch["data"], batch["labels"]
        inputs, labels = inputs.to(device), labels.to(device)

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
    inputs, labels = batch["data"], batch["labels"]
    inputs, labels = inputs.to(device), labels.to(device)

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    _, predicted = torch.max(outputs, 1)
    batch_total = labels.size(0)
    batch_correct = (predicted == labels).sum().item()
    return batch_correct, batch_total, loss.item()


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
    validation_max_batches=None,
    patience=10,
    time_limit=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric_path =os.path.join(model_folder, f'metrics_training_loop.pt')
    plot_path = os.path.join(model_folder, f'plots_training_loop.png')
    print(f"Using device: {device}")
    print(f"Using metrics path: {metric_path}")
    print(f"Using patience: {patience}")

    trainloader_len = _safe_len(trainloader)
    validationloader_len = _safe_len(validationloader)
    if trainloader_len is not None:
        print(f"Training batches per pass: {trainloader_len}")
    else:
        print("Training loader is streaming/iterable (unknown length per pass).")

    if validationloader_len is not None:
        print(f"Validation batches per pass: {validationloader_len}")
    else:
        print("Validation loader length is unknown.")

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

    print(f"Using max_train_steps: {max_train_steps}")
    print(f"Using steps_per_epoch: {steps_per_epoch}")
    print(f"Using eval_every_steps: {eval_every_steps}")
    print(f"Using log_every_steps: {log_every_steps}")

    # time limit in readable format for logging
    if time_limit:
        time_limit_str = f"{time_limit // 3600}h {(time_limit % 3600) // 60}m {time_limit % 60}s"
        print(f"Using time limit: {time_limit_str}.")
    print(f"Using plot path: {plot_path}")

    if device == torch.device("cpu"):
        print("WARNING: Training on CPU, this may be slow. Consider using a GPU for faster training.")
    model.to(device)

    best_model_state_dict = None
    epochs_best_model = None
    metrics = Metrics(patience=patience)

    start_time = time.time()

    global_step = 0
    train_iter = iter(trainloader)
    interval_correct = 0
    interval_total = 0
    interval_loss = 0.0
    interval_batches = 0
    interval_data_wait = 0.0
    interval_compute_time = 0.0
    interval_start = time.time()
    print("Starting training loop...")
    while global_step < max_train_steps:
        model.train()
        data_wait_start = time.time()
        # Get next batch, or restart the iterator if we've reached the end of the dataloader
        try:
            batch = next(train_iter)
        except StopIteration:
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

        compute_start = time.time()
        batch_correct, batch_total, batch_loss = _train_single_batch(model, batch, optimizer, criterion, device)
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
            avg_compute_ms = (interval_compute_time / max(interval_batches, 1)) * 1000
            total_interval_time = max(interval_data_wait + interval_compute_time, 1e-9)
            samples_per_sec = interval_total / total_interval_time
            data_wait_fraction = interval_data_wait / total_interval_time
            print(
                f"Step {global_step}/{max_train_steps} | "
                f"Train Acc: {running_acc * 100:.4f}% | "
                f"Train Loss: {running_loss:.4f} | "
                f"Avg Data Wait: {avg_data_wait_ms:.1f}ms | "
                f"Avg Compute: {avg_compute_ms:.1f}ms | "
                f"Samples/s: {samples_per_sec:.1f} | "
                f"Data Wait Share: {data_wait_fraction * 100:.1f}%"
            )
        batch = None  # Free batch memory
        should_eval = (global_step % eval_every_steps == 0) or (global_step == max_train_steps)
        if should_eval:
            model.eval()
            

            train_acc = interval_correct / max(interval_total, 1)
            train_loss = interval_loss / max(interval_batches, 1)
            # get the confusion matrix of the validation set predictions
            conf_matrix = confusion_matrix(
                y_true=[label for batch in validationloader for label in batch["labels"].numpy()],
                y_pred=[pred for batch in validationloader for pred in batch["predictions"].numpy()]
            )

            try:
                val_acc, val_loss = calculate_accuracy_and_loss(
                    model,
                    validationloader,
                    criterion,
                    device,
                    max_batches=validation_max_batches,
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

            print(f'Eval checkpoint at step {global_step} (pseudo-epoch {pseudo_epoch}):')
            print(f'\tTraining  \tAccuracy: {metrics.training_accuracy[-1]:.4f}%\tLoss: {metrics.train_loss[-1]:.4f}')
            print(f'\tValidation\tAccuracy: {metrics.validation_accuracy[-1]:.4f}%\tLoss: {metrics.validation_loss[-1]:.4f}')
            print('-' * 100)

            improved = metrics.model_improved()
            if improved:
                best_model_state_dict = model.state_dict()
                path = os.path.join(model_folder, f'current_best_model.pth')
                torch.save(best_model_state_dict, path)
                print(f"\tSaved best model at step {global_step} to {path}")
                epochs_best_model = pseudo_epoch
            else:
                metrics.patience_counter += 1

            print(f'\tPatience Counter: {metrics.patience_counter}/{patience}')
            print(
                f'\tTime for eval interval: {interval_time // 60:.2f}m {interval_time % 60:.0f}s\t'
                f'(avg: {average_time_per_eval // 60}m {average_time_per_eval % 60:.0f}s/check)\t'
                f'Time elapsed since start: {(time.time() - start_time) // 60:.0f}m'
            )
            print(
                f'\tPipeline profile: data_wait={interval_data_wait:.2f}s, '
                f'compute={interval_compute_time:.2f}s, '
                f'data_wait_share={100 * interval_data_wait / max(interval_data_wait + interval_compute_time, 1e-9):.1f}%'
            )
            if scheduler:
                print(f'\tLearning Rate: {optimizer.param_groups[0]["lr"]:.2e}')
            print('=' * 100)

            interval_correct = 0
            interval_total = 0
            interval_loss = 0.0
            interval_batches = 0
            interval_data_wait = 0.0
            interval_compute_time = 0.0
            interval_start = time.time()

            if metrics.patience_counter >= patience:
                print("Early stopping due to overfitting.")
                break

        if time_limit and (time.time() - start_time) > time_limit:
            print("Time limit reached, stopping training.")
            break

    if best_model_state_dict is None:
        best_model_state_dict = model.state_dict()
        epochs_best_model = (global_step - 1) // steps_per_epoch + 1 if global_step > 0 else 0

    print('Finished Training')
    return best_model_state_dict, epochs_best_model,metrics



