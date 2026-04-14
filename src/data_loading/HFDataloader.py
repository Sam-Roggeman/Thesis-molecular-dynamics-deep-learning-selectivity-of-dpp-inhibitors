import shutil

import datasets
import os
import numpy as np

import torch
from src.model_training.LabelEncoder import LabelEncoder
from src.model_training.batch_preprocessing import TARGET_PIXELS
from src.utils.cacheManager import cacheManager, construct_file_name
from src.utils.training_config import TrainingConfig

import resource


label_encoder = LabelEncoder()


def _is_host_oom_error(exc: BaseException) -> bool:
    """Return True for common CPU/host-memory OOM failures."""
    if isinstance(exc, MemoryError):
        return True

    oom_markers = (
        "out of memory",
        "cannot allocate memory",
        "can't allocate memory",
        "unable to allocate",
        "defaultcpuallocator",
        "std::bad_alloc",
        "bad allocation",
    )

    seen = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, MemoryError):
            return True
        msg = f"{type(current).__name__}: {current}".lower()
        if any(marker in msg for marker in oom_markers):
            return True
        current = current.__cause__ or current.__context__

    return False


def _is_map_worker_crash_error(exc: BaseException) -> bool:
    """Return True when Hugging Face map multiprocessing workers crash."""
    markers = (
        "one of the subprocesses has abruptly died during map operation",
        "a worker process managed by the executor was unexpectedly terminated",
        "brokenprocesspool",
    )

    seen = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        msg = f"{type(current).__name__}: {current}".lower()
        if any(marker in msg for marker in markers):
            return True
        current = current.__cause__ or current.__context__

    return False


def _encode_and_pack_batch(batch):
    labels = batch.get("labels")
    data = batch.get("data")

    if labels is None and data is None:
        return batch

    encoded = []
    if labels is not None:
        for label in labels:
            if isinstance(label, str):
                encoded.append(label_encoder.encode_label(label))
            else:
                encoded.append(int(label))

    if data is None:
        return {"labels": encoded}

    batch_size = len(data)
    packed = np.zeros((batch_size, TARGET_PIXELS, 3), dtype=np.float32)
    num_atoms = np.zeros((batch_size,), dtype=np.int16)
    for i, coords in enumerate(data):
        arr = np.asarray(coords, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 3)
        elif arr.ndim != 2 or arr.shape[1] != 3:
            arr = arr.reshape(-1, 3)

        n = min(arr.shape[0], TARGET_PIXELS)
        if n > 0:
            packed[i, :n, :] = arr[:n, :]
        num_atoms[i] = n

    result = {"data": packed, "num_atoms": num_atoms}
    if labels is not None:
        result["labels"] = encoded
    return result


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

def initialize_dataloaders(config: TrainingConfig, cache_manager: cacheManager, splits=None, keep_all_columns: bool = False, streaming: bool = False) -> DataLoaderDict:
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
    print("\tPre-encoding labels and packing coordinates into fixed-size tensors in dataset artifacts...")
    for split in splits:
        batch_size = max(1, int(config.transform_batch_size))
        num_workers = max(0, int(config.num_cpus))
        print(f"\t\tPreparing map for {split} split with batch size {batch_size} and num_workers {num_workers}...")
        while True:
            try: 
                used_percentage_str = f"{int(config.dataset_size * 100)}pct" 
                dataset_dict[split] = dataset_dict[split].map(
                    _encode_and_pack_batch,
                    batched=True,
                    batch_size=batch_size,
                    num_proc=max(1, num_workers),
                    desc=f"Encoding labels and packing coords for {split}",
                    cache_file_name=cache_manager.get_file_path(construct_file_name(used_percentage=used_percentage_str, splitname=split, prefix="labels_and_coords_packed",  extension="arrow"))
                )
                break
            except Exception as e:
                print(f"Error during map for {split} split with batch size {batch_size}: {e}")
                # If a host-memory OOM occurs during map, reduce batch size and retry.
                if( _is_host_oom_error(e) or _is_map_worker_crash_error(e) ) and batch_size > 1:
                    print(f"Host-memory OOM during map for {split} split with batch size {batch_size}. Reducing batch size and retrying...")
                    batch_size = max(1, (3*batch_size) // 4)
                    print(f"\tNew batch size: {batch_size}")
                    continue
                else:
                    raise
    print("\tDataset map preprocessing done; using fast fixed-shape batch path.")
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
    print("\tEncoding labels and packing coordinates in streaming pipeline...")
    for split in splits:
        if split in dataset_dict:
            batch_size = max(1, int(config.transform_batch_size))
            num_workers = max(0, int(config.num_cpus))
            print(f"\t\tPreparing streaming map for {split} split with batch size {batch_size} and num_workers {num_workers}...")
            while True:

                try:

                    dataset_dict[split] = dataset_dict[split].map(
                        _encode_and_pack_batch,
                        batched=True,
                        batch_size=batch_size,
                        num_proc=max(1, num_workers),
                        desc=f"Encoding labels and packing coords for {split} (streaming)",
                    )
                    break
                # if out of memory error occurs during map, reduce batch size
                except Exception as e:
                    if _is_host_oom_error(e) and batch_size > 1:
                        print(f"Host-memory OOM during streaming map for {split} split with batch size {batch_size}. Reducing batch size and retrying...")
                        batch_size = max(1, (3*batch_size) // 4)
                        print(f"\tNew batch size: {batch_size}")
                        continue
                    else:
                        raise
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
    
    