import shutil

import datasets
import os

import torch
from src.utils.cacheManager import cacheManager, construct_file_name
from src.utils.training_config import TrainingConfig

import resource


def _collate_raw_batch(batch):
    """Keep variable-length fields as python lists for on-the-fly preprocessing."""
    if not batch:
        return {}
    keys = batch[0].keys()
    return {key: [sample.get(key) for sample in batch] for key in keys}

def _download_dataset(config: TrainingConfig, splits=None) -> datasets.DatasetDict:
    print("Downloading dataset...")
    dataset_size = config.dataset_size
    assert dataset_size > 0 and dataset_size <= 1, "Dataset size must be between 0 and 1"
    print(f"\tDownloading {dataset_size} of {config.dataset_location}")
    # download only a subset of the dataset if dataset_size < 1
    split_size = int(dataset_size * 100)
    split_str = f"[:{split_size}%]" if dataset_size < 0.9999 else ""
    cache_dir = os.environ.get("HF_DOWNLOADED_DATASET_DIR")

    if splits is not None:
        print(f"\t\tOnly downloading splits: {splits}")
        _split_arg = {split: f"{split}{split_str}" for split in splits}
    else:
        for split in ['train', 'validation', 'test']:
            print(f"\t\tDownloading {split_size}% of {split} split")
        _split_arg = {
            "train": f"train{split_str}",
            "validation": f"validation{split_str}",
            "test": f"test{split_str}",
        }    
    # cache dir
    print(f"\tUsing cache directory: {cache_dir}")
    dataset_dict: datasets.DatasetDict = datasets.load_dataset(
        config.dataset_location,
        split=_split_arg,
        cache_dir=os.environ.get("HF_DOWNLOADED_DATASET_DIR"),
        token=os.environ.get("HF_TOKEN"),
        num_proc= config.num_cpus
    )
    print("\t...downloading_dataset complete")
    # wrap the dataset in a DatasetDict if it's not already one
    if not isinstance(dataset_dict, datasets.DatasetDict):
        dataset_dict = datasets.DatasetDict({splits[0]: dataset_dict})
    return dataset_dict

def _download_streaming_dataset(config: TrainingConfig, splits: list = ["train", "validation", "test"],  shuffle: bool = False) -> datasets.IterableDatasetDict:
    dataset_size = config.dataset_size
    assert dataset_size > 0 and dataset_size <= 1, "Dataset size must be between 0 and 1"
    print(f"\tDownloading {dataset_size} of {config.dataset_location}")
    dataset: datasets.IterableDatasetDict = datasets.load_dataset(
        config.dataset_location,
        token=os.environ.get("HF_TOKEN"),
        streaming=True
    )
    if shuffle:
        for split in splits:
            if split != "train":
                print(f"\tShuffling {split} split...")
                dataset[split] = dataset[split].shuffle(seed=config.shuffle_seed, buffer_size=config.shuffle_buffer_size)
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
    # take only the specified splits
    return datasets.IterableDatasetDict({split: dataset[split] for split in splits if split in dataset})   

# Define a type for the dataloader dict
DataLoaderDict = dict[str, torch.utils.data.DataLoader]

def initialize_dataloaders(config: TrainingConfig, cache_manager: cacheManager,splits=None, keep_all_columns: bool = False, streaming: bool = False) -> DataLoaderDict:
    """
    Initialize the dataloader for training.
    """
    if splits is None:
        splits = ['train', 'validation', 'test']
    if streaming:
        return initialize_streaming_dataloader(config, keep_all_columns=keep_all_columns, splits=splits)

    # Download the dataset
    print("Initializing dataloader...")
    print("\tDownloading dataset...")
    dataset_dict: datasets.DatasetDict = _download_dataset(config, splits=splits)
    
    print("\t...initializing_dataloader complete")
    dataset_dict = dataset_dict.rename_columns({'coordinates': 'data', 'binding_type': 'labels'})
    print("\tSkipping dataset.map transforms; using on-the-fly batch preprocessing.")
    if not keep_all_columns:
        drop_cols = [
            c for c in ['pdb_id', 'ligand_name', 'replica_id']
            if c in dataset_dict[splits[0]].column_names
        ]
        for split in splits:
            dataset_dict[split] = dataset_dict[split].remove_columns(drop_cols)

    dataloader_workers = max(0, config.num_cpus)
    dataloader_args = {
        "batch_size": config.batch_size,
        "num_workers": dataloader_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": dataloader_workers > 0,
        "collate_fn": _collate_raw_batch,
    }
    if dataloader_workers > 0:
        dataloader_args["prefetch_factor"] = 2
    resource.setrlimit(resource.RLIMIT_NOFILE, (10810, 10810))
    
    dataloader_dict = {}
    for split in splits:
        print(f"\t\tCreating dataloader for {split} split with batch size {config.batch_size} and num_workers {config.num_cpus}...")
        dataloader = torch.utils.data.DataLoader(dataset_dict[split], **dataloader_args)
        dataloader_dict[split] = dataloader
    return dataloader_dict

def initialize_streaming_dataloader(config: TrainingConfig, keep_all_columns: bool = False, splits: list = ["train", "validation", "test"], shuffle: bool = False) -> DataLoaderDict:
    """
    Initialize the streaming dataloader for training.
    """
    # Initialize the streaming dataloader
    print("Initializing streaming dataloader...")
    dataset_dict: datasets.IterableDatasetDict = _download_streaming_dataset(config, splits=splits, shuffle=shuffle)

    dataset_dict = dataset_dict.rename_columns({'coordinates': 'data', 'binding_type': 'labels'})
    if not keep_all_columns:
        drop_cols = ['pdb_id', 'ligand_name']
        for split in splits:
            if split in dataset_dict:
                existing = [c for c in drop_cols if c in dataset_dict[split].column_names]
                if existing:
                    dataset_dict[split] = dataset_dict[split].remove_columns(existing)

    dataloader_dict = {}
    for split in splits:
        print(f"\t\tPreparing dataloader for {split} split (on-the-fly preprocessing)...")
        dataloader_workers = max(0, config.num_cpus)
        dataloader_args = {
            "batch_size": config.batch_size,
            "num_workers": dataloader_workers,
            "pin_memory": True,
            "persistent_workers": dataloader_workers > 0,
            "collate_fn": _collate_raw_batch,
        }
        if dataloader_workers > 0:
            dataloader_args["prefetch_factor"] = 2
        dataloader = torch.utils.data.DataLoader(dataset_dict[split], **dataloader_args)
        dataloader_dict[split] = dataloader
    print("\t...streaming dataloaders ready")

    resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))
    
    print("\t...initializing_streaming_dataloader complete")
    return dataloader_dict

def clear_cache(remove_dataset_cache: bool = False, remove_mapped_cache: bool = True):
    """
    Clear the Hugging Face dataset cache directory.
    """
    cache_dir = os.environ.get("HF_DOWNLOADED_DATASET_DIR")
    mapped_cache_dir = os.environ.get("HF_MAPPED_DATASET_DIR")

    if remove_dataset_cache and os.path.exists(cache_dir):
        print(f"Clearing dataset cache directory: {cache_dir}")
        shutil.rmtree(cache_dir)
        print("\t...dataset cache cleared")

    if remove_mapped_cache and os.path.exists(mapped_cache_dir):
        print(f"Clearing mapped cache directory: {mapped_cache_dir}")
        shutil.rmtree(mapped_cache_dir)
        print("\t...mapped cache cleared")
    else:
        print(f"Cache directory {cache_dir} does not exist, nothing to clear.")



if __name__ == "__main__":
    import argparse
    # load and cache the dataset and dataloader
    # read command line arguments for dataset size and batch size
    parser = argparse.ArgumentParser(description="Prepare cache for Hugging Face dataset and dataloader")
    parser.add_argument("--dataset-size", type=int, default=100, help="Size of the dataset to use for training.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--cpus", type=int, default=1, help="Number of CPUs to use for training.")
    parser.add_argument("--use-all-columns", action="store_true", help="Whether to keep all columns in the dataset or remove extra columns.")
    
    args = parser.parse_args()
    config = TrainingConfig(    
        dataset_location="Sam-Roggeman/SamRoggeman_Thesis_Dataset",
        dataset_size=args.dataset_size / 100,
        batch_size=args.batch_size,
        transform_batch_size=128,
        num_cpus=args.cpus,
        
    )
    
        
    dataloader_dict = initialize_dataloaders(config, keep_all_columns=args.use_all_columns)
    print("Dataloaders initialized and cached successfully.")
    
    