from typing import Tuple, Dict
import os
from datetime import datetime
import torch
from torch import optim
from huggingface_hub import HfApi
import sys
from src.model_training.metric_functions import calculate_accuracy
from src.model_training.utils import get_device, get_subset, training_loop
from src.utils.training_config import TrainingConfig
from src.utils.logger import setup_logger, replace_output
import datasets
import torch.nn as nn


def setup_directories_and_logging(config: TrainingConfig, model_name: str) -> Tuple[str, object]:
    """Create model directory and setup logging"""
    time_string = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir = f'./models/{config.model_class.__name__}/{model_name}/{time_string}/'
    os.makedirs(model_dir, exist_ok=True)

    logger = setup_logger(
        log_file="outputlog.txt",
        log_dir=model_dir,
        logging_enabled=True,
        console_enabled=False
    )
    replace_output(logger)

    return model_dir, logger


def save_results(model_state_dict, model_dir: str, model_name: str, metrics, test_acc: float):
    """Save model, metrics, and test accuracy"""
    filepath = os.path.join(model_dir, f"{model_name}.pth")
    torch.save(model_state_dict, filepath)
    print(f"Model saved to: {filepath}")

    plot_path = filepath.replace('.pth', '.png')
    metric_path = filepath.replace('.pth', '.metrics')
    metrics.save_plot("Model Performance", plot_path)
    metrics.save_metrics(metric_path)

    print(f"Plot saved to: {plot_path}")
    print(f"Test Accuracy: {test_acc:.4f}")

def save_config(config: TrainingConfig, model_dir: str):
    """Save training configuration to a file"""
    config_path = os.path.join(model_dir, "training_config.txt")
    with open(config_path, 'w') as f:
        for field in config.__dataclass_fields__:
            value = getattr(config, field)
            f.write(f"{field}: {value}\n")
    print(f"Training configuration saved to: {config_path}")
def is_online_dataset(dataset_location) -> bool:
    """Check if the dataset location is an online dataset (Hugging Face Hub)"""
    return not os.path.exists(dataset_location)
def prepare_dataset(dataset_train, dataset_val, dataset_test, config: TrainingConfig) -> Dict:
    """Subset and transform datasets"""
    device = get_device()

    dss = {"train": dataset_train, "val": dataset_val, "test": dataset_test}
    for dataset_key in dss:
        dataset = dss[dataset_key]
        # Only rename if the original column names exist
        if "coordinates" in dataset.column_names:
            dataset = dataset.rename_column("coordinates", "data")
        if "binding_type" in dataset.column_names:
            dataset = dataset.rename_column("binding_type", "labels")
        dss[dataset_key] = dataset
    # Apply transforms
    dss["train"] = dss["train"].map(
        config.training_transorm,
        batch_size=config.transform_batch_size,
        batched=True,
        input_columns=['data', 'labels', "num_atoms"],
        remove_columns=['pdb_id', 'dpp_class', 'ligand_name', 'num_atoms'],
        num_proc=8,
        cache_file_name=os.path.join(config.cache_folder, f"train_transformed_{config.dataset_size}.arrow"),
    )

    for split in ["val", "test"]:
        dss[split] = dss[split].map(
            config.validation_transform,
            batch_size=config.transform_batch_size,
            batched=True,
            input_columns=['data', 'labels', "num_atoms"],
            remove_columns=['pdb_id', 'dpp_class', 'ligand_name', 'num_atoms'],
            num_proc=config.transform_num_proc,
            cache_file_name=os.path.join(config.cache_folder, f"{split}_transformed_{config.dataset_size}.arrow"),
        )

    # Set format and create dataloaders
    for split in ["train", "val", "test"]:
        dss[split].set_format(type='torch', columns=['data', 'labels'], device=device)

    dataloaders = {
        "train": torch.utils.data.DataLoader(dss["train"], batch_size=config.batch_size, shuffle=True, num_workers=8),
        "val": torch.utils.data.DataLoader(dss["val"], batch_size=config.batch_size, shuffle=False, num_workers=8),
        "test": torch.utils.data.DataLoader(dss["test"], batch_size=config.batch_size, shuffle=False, num_workers=8),
    }

    return dataloaders

def load_and_prepare_facehub_datasets(config: TrainingConfig) -> Dict:
    """Load datasets from Hugging Face Hub, subset, and transform"""
    device = get_device()
    num_proc_load = 1 if "pydevd" in sys.modules else 8
    # set size in string format
    percent_str = str(int(config.dataset_size * 100))
    cache_folder = os.path.join(config.cache_folder)
    cached = os.path.exists(cache_folder)
    # Load from Hugging Face Hub
    dataset_train = datasets.load_dataset(config.dataset_location, split=f"train[:{percent_str}%]", token=config.hf_token, num_proc=num_proc_load, cache_dir=cache_folder)
    dataset_val = datasets.load_dataset(config.dataset_location, split=f"validation[:{percent_str}%]", token=config.hf_token, num_proc=num_proc_load, cache_dir=cache_folder)
    dataset_test = datasets.load_dataset(config.dataset_location, split=f"test[:{percent_str}%]", token=config.hf_token, num_proc=num_proc_load, cache_dir=cache_folder)
    if cached:
        dss = {"train": dataset_train, "val": dataset_val, "test": dataset_test}
        for dataset_key in dss:
            dss[dataset_key].set_format(type='torch', columns=['data', 'labels'], device=device)

        dataloaders = {
            "train": torch.utils.data.DataLoader(dss["train"], batch_size=config.batch_size, shuffle=True, num_workers=num_proc_load),
            "val": torch.utils.data.DataLoader(dss["val"], batch_size=config.batch_size, shuffle=False, num_workers=num_proc_load),
            "test": torch.utils.data.DataLoader(dss["test"], batch_size=config.batch_size, shuffle=False, num_workers=num_proc_load),

        }
        return dataloaders

    return prepare_dataset(dataset_train, dataset_val, dataset_test, config)

def load_and_prepare_datasets(config: TrainingConfig) -> Dict:
    """Load datasets from disk, subset, and transform"""
    # Load from disk
    dataset_train = datasets.load_from_disk(os.path.join(config.dataset_location, "train"))
    dataset_val = datasets.load_from_disk(os.path.join(config.dataset_location, "val"))
    dataset_test = datasets.load_from_disk(os.path.join(config.dataset_location, "test"))

    return prepare_dataset(dataset_train, dataset_val, dataset_test, config)


def train_model(config: TrainingConfig, model_name: str):
    """Main training function - single entry point for all models"""

    # Setup
    run_dir, logger = setup_directories_and_logging(config, model_name)
    print(f"Training {config.model_class.__name__}. Model will be saved to: {run_dir}")
    print(f"Weight Decay: {config.weight_decay}, Learning Rate: {config.learning_rate}")
    print(f"Max Num Epochs: {config.max_nr_epochs}")
    save_config(config, run_dir)

    # Initialize model
    model = config.model_class(**config.model_args)

    # Load and prepare data
    dataloaders = load_and_prepare_facehub_datasets(config)

    # Setup training
    device = get_device()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    # Train
    model_state_dict, nr_epochs, metrics = training_loop(
        model=model,
        model_folder=run_dir,
        trainloader=dataloaders["train"],
        validationloader=dataloaders["val"],
        optimizer=optimizer,
        criterion=criterion,
        max_epochs=config.max_nr_epochs,
        patience=config.patience,
        time_limit=config.time_limit
    )

    model.load_state_dict(model_state_dict)

    # Evaluate and save
    test_acc = calculate_accuracy(model, dataloaders["test"], device=device)
    save_results(model_state_dict, run_dir, model_name, metrics, test_acc)
    # cleanup
    del model
    torch.cuda.empty_cache()
    print("Training complete.")

