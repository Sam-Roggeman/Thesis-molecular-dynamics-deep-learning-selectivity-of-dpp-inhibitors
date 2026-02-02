from typing import Tuple, Dict, Any
import os
from datetime import datetime
import torch
from torch import optim
from huggingface_hub import HfApi
import sys

from torch.utils.data import DataLoader

from src.model_training.metric_functions import all_statistics
from src.model_training.utils import get_device, get_subset, training_loop
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


def save_results(model_state_dict, model_dir: str, model_name: str, metrics, stats: dict[str, float]):
    """Save model, metrics, and test accuracy"""
    filepath = os.path.join(model_dir, f"{model_name}.pth")
    torch.save(model_state_dict, filepath)
    print(f"Model saved to: {filepath}")

    plot_path = filepath.replace('.pth', '.png')
    metric_path = filepath.replace('.pth', '.metrics')
    metrics.save_plot("Model Performance", plot_path)
    metrics.save_metrics(metric_path)

    print(f"Plot saved to: {plot_path}")
    for stat_name, stat_value in stats.items():
        print(f"{stat_name}: {stat_value}")

def rename_columns(dataset) -> datasets.Dataset:
    """Rename dataset columns to standard names 'data' and 'labels'"""
    # Only rename if the original column names exist
    if "coordinates" in dataset.column_names:
        dataset = dataset.rename_column("coordinates", "data")
    if "binding_type" in dataset.column_names:
        dataset = dataset.rename_column("binding_type", "labels")
    return dataset
def set_format_and_create_dataloaders(dataset, config: TrainingConfig, device):
    dataset.set_format(type='torch', columns=['data', 'labels'], device=device)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=8)
    return dataloader

def prepare_dataset(dataset, transform, split_name, config: TrainingConfig, cache_dir) -> DataLoader[Any]:
    """Subset and transform datasets"""
    device = get_device()

    dataset = rename_columns(dataset)
    # Apply transforms
    dataset = dataset.map(
        transform,
        batch_size=config.transform_batch_size,
        batched=True,
        input_columns=['data', 'labels', "num_atoms"],
        remove_columns=['pdb_id', 'dpp_class', 'ligand_name', 'num_atoms'],
        num_proc=8,
        cache_file_name=os.path.join(cache_dir, f"{split_name}_transformed_{config.dataset_size}.arrow"),
    )
    # Set format and create dataloaders
    return set_format_and_create_dataloaders(dataset, config, device)
def load_and_prepare_test(config) -> DataLoader[Any]:
    """Load only validation and test datasets from huggingface hub, subset, and transform"""
    cache_folder = os.environ.get("HF_CACHE_DIR")
    # Load from Hugging Face Hub
    mapped_cache_folder = os.path.join(cache_folder, "mapped_cache")
    dataset_val = load_dataset_from_hf(config, "test")

    return prepare_dataset(dataset_val, config.validation_transform, "test", config, mapped_cache_folder)

def load_dataset_from_hf(config: TrainingConfig, split: str):
    """Load dataset from Hugging Face Hub"""
    num_proc_load = 1 if "pydevd" in sys.modules else 8
    percent_str = str(int(config.dataset_size * 100))
    cache_folder = os.environ.get("HF_CACHE_DIR")
    hf_token = os.environ.get("HF_TOKEN")
    # set size in string format
    downloaded_cache_folder = os.path.join(cache_folder, "downloaded_cache")
    return datasets.load_dataset(config.dataset_location, split=f"{split}[:{percent_str}%]", token=hf_token, num_proc=num_proc_load, cache_dir=downloaded_cache_folder)

def load_and_prepare_facehub_datasets(config: TrainingConfig) -> Dict:
    """Load datasets from Hugging Face Hub, subset, and transform"""
    # Load from Hugging Face Hub
    cache_folder = os.environ.get("HF_CACHE_DIR")
    mapped_cache_folder = os.path.join(cache_folder, "mapped_cache")
    dataset_train = load_dataset_from_hf(config, "train")
    dataset_val = load_dataset_from_hf(config, "validation")
    dataset_test = load_dataset_from_hf(config, "test")

    return {
        "train": prepare_dataset(dataset_train, config.training_transorm, "train", config, mapped_cache_folder),
        "val": prepare_dataset(dataset_val, config.validation_transform, "val", config, mapped_cache_folder),
        "test": prepare_dataset(dataset_test, config.validation_transform, "test", config, mapped_cache_folder)
    }




def train_model(config: TrainingConfig, model_name: str):
    """Main training function - single entry point for all models"""

    # Setup
    run_dir, logger = setup_directories_and_logging(config, model_name)
    print(f"Training {config.model_class.__name__}. Model will be saved to: {run_dir}")
    print(f"Weight Decay: {config.weight_decay}, Learning Rate: {config.learning_rate}")
    print(f"Max Num Epochs: {config.max_nr_epochs}")
    config.save(os.path.join(run_dir))

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
    statistics = all_statistics(model, dataloaders["test"], device, criterion)

    save_results(model_state_dict, run_dir, model_name, metrics, statistics)
    # cleanup
    del model
    torch.cuda.empty_cache()
    print("Training complete.")

