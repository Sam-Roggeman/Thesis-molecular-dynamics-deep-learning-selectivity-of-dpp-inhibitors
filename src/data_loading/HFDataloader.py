import shutil

import datasets
import os

import torch

from src.utils.training_config import TrainingConfig

import resource

def _download_dataset(config: TrainingConfig, splits=None) -> datasets.DatasetDict:
    print("Downloading dataset...")
    dataset_size = config.dataset_size
    assert dataset_size > 0 and dataset_size <= 1, "Dataset size must be between 0 and 1"
    print(f"\tDownloading {dataset_size} of {config.dataset_location}")
    # download only a subset of the dataset if dataset_size < 1
    split_size = int(dataset_size * 100)

    if splits is not None:
        print(f"\t\tOnly downloading splits: {splits}")
        _split_arg = {split: f"{split}[:{split_size}%]" for split in splits}
    else:
        for split in ['train', 'validation', 'test']:
            print(f"\t\tDownloading {split_size}% of {split} split")
        _split_arg = {
            "train": f"train[:{split_size}%]",
            "validation": f"validation[:{split_size}%]",
            "test": f"test[:{split_size}%]",
        }        
    dataset_dict: datasets.DatasetDict = datasets.load_dataset(
        config.dataset_location,
        split=_split_arg,
        cache_dir=os.environ.get("HF_DOWNLOADED_DATASET_DIR"),
        token=os.environ.get("HF_TOKEN"),
        num_proc= config.num_cpus
    )
    print("\t...downloading_dataset complete")
    # create a dataset dict with the three splits and return it    
    return dataset_dict

def _download_streaming_dataset(config: TrainingConfig) -> datasets.IterableDatasetDict:
    dataset_size = config.dataset_size
    assert dataset_size > 0 and dataset_size <= 1, "Dataset size must be between 0 and 1"
    print(f"\tDownloading {dataset_size} of {config.dataset_location}")
    dataset: datasets.IterableDatasetDict = datasets.load_dataset(
        config.dataset_location,
        token=os.environ.get("HF_TOKEN"),
        streaming=True
    )
    # shuffle the trainssplit of the dataset
    dataset["train"] = dataset["train"].shuffle(seed=config.shuffle_seed, buffer_size=config.shuffle_buffer_size)
    if dataset_size < 1:
        splitinfo: datasets.DatasetInfo = dataset["train"].info
        for split in dataset.keys():
            try:
                total_samples = splitinfo.splits[split].num_examples
                n_samples = int(total_samples * config.dataset_size)
                n_samples = max(config.batch_size, n_samples)
                n_samples = ((n_samples + config.batch_size - 1) // config.batch_size) * config.batch_size
                print(f"\tUsing streaming subset for {split}: {n_samples}/{total_samples} samples")
                dataset[split] = dataset[split].take(n_samples)
            except Exception:
                print(
                    f"Warning: Could not determine split size for streaming subset on {split}. "
                    "Falling back to full streamed split."
                )
    print("\t...downloading_streaming_dataset complete")
    return dataset

# Define a type for the dataloader dict
DataLoaderDict = dict[str, torch.utils.data.DataLoader]

def initialize_dataloaders(config: TrainingConfig, splits=None) -> DataLoaderDict:
    """
    Initialize the dataloader for training.
    """
    # Download the dataset
    print("Initializing dataloader...")
    print("\tDownloading dataset...")
    dataset_dict: datasets.DatasetDict = _download_dataset(config, splits=splits)
    print("\t...initializing_dataloader complete")
    dataset_dict = dataset_dict.rename_columns({'coordinates': 'data', 'binding_type': 'labels'})
    # apply the training transform to the training split and the validation transform to the validation and test splits
    
    print("\tApplying transforms...")
    map_args = {
        "batched": True, 
        "batch_size": config.transform_batch_size, 
        "num_proc": config.num_cpus,
        "input_columns": ['data', 'labels', "num_atoms"], 
        "remove_columns": ['pdb_id', 'dpp_class', 'ligand_name', 'num_atoms', 'replica_id']
        }
    mapped_cache_dir = os.path.join(os.environ.get("HF_CACHE_DIR"), "mapped_datasets")
    # copy the cache dir to the faster local storage if running in GPULAB
    if "GPULAB_CPUS_RESERVED" in os.environ:
        old_cache_dir = mapped_cache_dir
        mapped_cache_dir = os.path.join("/project_scratch/dataset_cache/", "mapped_datasets")
        # copy the cache dir to the faster local storage if it doesn't already exist there
        for file in os.listdir(old_cache_dir):
            print(f"Copying mapped dataset cache from {old_cache_dir} to {mapped_cache_dir} for faster access...")
            if file.endswith(".arrow"):
                fp = os.path.join(mapped_cache_dir, file)
                if not os.path.exists(fp):
                    shutil.copy(os.path.join(old_cache_dir, file), fp)
            print(f"\t...copying complete")    
                
    print(f"\tUsing mapped dataset cache directory: {mapped_cache_dir}")
    dataset_dict["train"] = dataset_dict["train"].map(
        config.training_transform,
        **map_args,
        load_from_cache_file=True,
        cache_file_name=os.path.join(mapped_cache_dir, f"train_transformed_{config.dataset_size * 100:.0f}pct.arrow"),
    )
    dataset_dict["validation"] = dataset_dict["validation"].map(
        config.validation_transform,
        **map_args,
        load_from_cache_file=True,
        cache_file_name=os.path.join(mapped_cache_dir, f"validation_transformed_{config.dataset_size * 100:.0f}pct.arrow"),
    )
    dataset_dict["test"] = dataset_dict["test"].map(
        config.validation_transform,
        **map_args,
        load_from_cache_file=True,
        cache_file_name=os.path.join(mapped_cache_dir, f"test_transformed_{config.dataset_size * 100:.0f}pct.arrow"),
    )
    print("\t...applying_transforms complete")
    dataloader_args = {
        "batch_size": config.batch_size,
        "num_workers": config.num_cpus,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": True,
        "prefetch_factor": 4, 
    }
    resource.setrlimit(resource.RLIMIT_NOFILE, (10810, 10810))
    
    dataset_dict = dataset_dict.with_format(type="torch", columns=["data", "labels"])
    train_dataloader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(dataset_dict["train"], **dataloader_args)
    validation_dataloader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(dataset_dict["validation"], **dataloader_args)
    test_dataloader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(dataset_dict["test"], **dataloader_args)
    return {"train": train_dataloader, "validation": validation_dataloader, "test": test_dataloader}

def initialize_streaming_dataloader(config: TrainingConfig) -> DataLoaderDict:
    """
    Initialize the streaming dataloader for training.
    """
    # Initialize the streaming dataloader
    print("Initializing streaming dataloader...")
    dataset_dict: datasets.IterableDatasetDict = _download_streaming_dataset(config)
    
    # Shuffle the training split
    dataset_dict["train"] = dataset_dict["train"].shuffle(seed=config.shuffle_seed, buffer_size=config.shuffle_buffer_size)
    dataset_dict = dataset_dict.rename_columns({'coordinates': 'data', 'binding_type': 'labels'})
    
    # apply the training transform to the training split and the validation transform to the validation and test splits
    map_args = {"batched": True, "batch_size": config.transform_batch_size, 
                "input_columns": ['data', 'labels', "num_atoms"], 
                "remove_columns": ['pdb_id', 'dpp_class', 'ligand_name', 'num_atoms']}
    dataset_dict["train"] = dataset_dict["train"].map(config.training_transform, **map_args)
    dataset_dict["validation"] = dataset_dict["validation"].map(config.validation_transform, **map_args)
    dataset_dict["test"] = dataset_dict["test"].map(config.validation_transform, **map_args)
    
    # set the format of the dataset to torch tensors and create dataloaders for each split
    dataset_dict = dataset_dict.with_format(type="torch")

    dataloader_args = {"batch_size": config.batch_size, "num_workers": config.num_cpus, "pin_memory": True}    
    train_dataloader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(dataset_dict["train"], **dataloader_args)
    validation_dataloader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(dataset_dict["validation"], **dataloader_args)
    test_dataloader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(dataset_dict["test"], **dataloader_args)
    dl_dict = {"train": train_dataloader, "validation": validation_dataloader, "test": test_dataloader}
    resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))
    
    print("\t...initializing_streaming_dataloader complete")
    return dl_dict
