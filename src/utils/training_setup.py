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
from src.data_postprocessing.model_testing import model_testing
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

def rename_columns(dataset) -> datasets.Dataset:
    """Rename dataset columns to standard names 'data' and 'labels'"""
    # Only rename if the original column names exist
    if "coordinates" in dataset.column_names:
        dataset = dataset.rename_column("coordinates", "data")
    if "binding_type" in dataset.column_names:
        dataset = dataset.rename_column("binding_type", "labels")
    return dataset
def set_format_and_create_dataloaders(dataset, config: TrainingConfig):
    is_iterable = isinstance(dataset, datasets.IterableDataset)
    num_workers = 0 if ("pydevd" in sys.modules or is_iterable) else config.num_cpus

    if is_iterable:
        # HF datasets versions differ: some IterableDataset.with_format versions
        # do not accept the `columns` argument.
        try:
            dataset = dataset.with_format(type='torch', columns=['data', 'labels'])
        except TypeError:
            dataset = dataset.with_format(type='torch')
        
    else:
        dataset.set_format(type='torch', columns=['data', 'labels'])

    dataloader_kwargs = {
        "batch_size": config.batch_size,
        "shuffle": not is_iterable,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        dataloader_kwargs.update({
            "persistent_workers": True,
            "prefetch_factor": 4,
            'multiprocessing_context': 'spawn'
        })

    dataloader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

    
    return dataloader

def prepare_dataset(dataset, transform, split_name, config: TrainingConfig, cache_dir) -> DataLoader[Any]:
    """Subset and transform datasets"""
    is_iterable = isinstance(dataset, datasets.IterableDataset)
    num_proc = 1 if ("pydevd" in sys.modules or is_iterable) else config.num_cpus

    dataset = rename_columns(dataset)
    if is_iterable:
        print(f"Applying transforms to {split_name} dataset (streaming, size unknown)...")
    else:
        print(f"Applying transforms to {split_name} dataset with {len(dataset)} samples...")
    # Apply transforms
    map_kwargs = {
        "batch_size": config.transform_batch_size,
        "batched": True,
        "input_columns": ['data', 'labels', "num_atoms"],
        "remove_columns": ['pdb_id', 'dpp_class', 'ligand_name', 'num_atoms'],
    }
    if not is_iterable:
        map_kwargs["num_proc"] = num_proc
        map_kwargs["cache_file_name"] = os.path.join(cache_dir, f"{split_name}_transformed_{config.dataset_size}.arrow")
    print(f"\tMapping {split_name} dataset with batch size {config.transform_batch_size} and num_proc={map_kwargs.get('num_proc', 'N/A')}...")
    dataset = dataset.map(transform, **map_kwargs)
    # Set format and create dataloaders
    print(f"\t Finished applying transforms to {split_name} dataset.")
    
    return set_format_and_create_dataloaders(dataset, config)
def load_and_prepare_test(config) -> DataLoader[Any]:
    """Load only validation and test datasets from huggingface hub, subset, and transform"""
    cache_folder = os.environ.get("HF_CACHE_DIR")
    # Load from Hugging Face Hub
    mapped_cache_folder = os.path.join(cache_folder, "mapped_cache")
    dataset_val = load_dataset_from_hf(config, "test")

    return prepare_dataset(dataset_val, config.validation_transform, "test", config, mapped_cache_folder)

def load_dataset_from_hf(config: TrainingConfig, split: str):
    """Load dataset from Hugging Face Hub"""
    hf_token = os.environ.get("HF_TOKEN")
    should_stream = (
        (split == "train" and config.stream_train_split)
        or (split == "validation" and config.stream_validation_split)
        or (split == "test" and config.stream_test_split)
    )
    print(f"Loading {split} split from Hugging Face Hub with streaming={should_stream}...")         
    if should_stream:
        dataset = datasets.load_dataset(
            config.dataset_location,
            split=split,
            token=hf_token,
            streaming=True,
        )
        if split == "train":
            dataset = dataset.shuffle(buffer_size=config.shuffle_buffer_size, seed=config.shuffle_seed)

        if config.dataset_size < 1.0:
            try:
                total_samples = dataset.info.splits[split].num_examples
                n_samples = int(total_samples * config.dataset_size)
                n_samples = max(config.batch_size, n_samples)
                n_samples = ((n_samples + config.batch_size - 1) // config.batch_size) * config.batch_size
                print(f"Using streaming subset for {split}: {n_samples}/{total_samples} samples")
                dataset = dataset.take(n_samples)
            except Exception:
                print(
                    f"Warning: Could not determine split size for streaming subset on {split}. "
                    "Falling back to full streamed split."
                )
        print(f"Finished loading {split} split from Hugging Face Hub with streaming={should_stream}.")
        return dataset

    split_spec = split
    if config.dataset_size < 1.0:
        percentage = config.dataset_size * 100
        split_spec = f"{split}[:{percentage}%]"

    return datasets.load_dataset(
        config.dataset_location,
        split=split_spec,
        token=hf_token,
        streaming=False,
    )

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



