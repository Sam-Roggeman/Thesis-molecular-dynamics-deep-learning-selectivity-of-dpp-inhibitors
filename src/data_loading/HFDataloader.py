import multiprocessing as mp

from src.data_loading.SafetensorsDataset import ShardedSafetensorsDataset
mp.set_start_method('spawn', force=True)
import shutil

import datasets
import os
import numpy as np
from safetensors.torch import save_file
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from src.model_training.LabelEncoder import LabelEncoder
from src.utils.cacheManager import cacheManager, construct_cache_identifier
from src.utils.training_config import TrainingConfig
import src.utils.logger as logging


def _effective_worker_count(requested_cpus: int) -> int:
    """Use physical-core-like worker count on hyperthreaded systems."""
    requested = max(1, int(requested_cpus))
    return max(1, requested // 2)


def _download_streaming_dataset(config: TrainingConfig, splits: list = ["train", "validation", "test"],  shuffle: bool = False) -> datasets.IterableDatasetDict:
    dataset_size = config.dataset_size
    assert dataset_size > 0 and dataset_size <= 1, "Dataset size must be between 0 and 1"
    logging.info(f"\tDownloading {dataset_size} of {config.dataset_location}")
    dataset: datasets.IterableDatasetDict = datasets.load_dataset(
        config.dataset_location,
        token=os.environ.get("HF_TOKEN"),
        streaming=True
    )
    if shuffle:
        for split in splits:
            if split != "train":
                logging.info(f"\tShuffling {split} split...")
                dataset[split] = dataset[split].shuffle(seed=config.shuffle_seed, buffer_size=config.shuffle_buffer_size)
    if dataset_size < 1:
        splitinfo: datasets.DatasetInfo = dataset["train"].info
        for split in dataset.keys():
            try:
                total_samples = splitinfo.splits[split].num_examples
                n_samples = int(total_samples * config.dataset_size)
                n_samples = max(config.batch_size, n_samples)
                n_samples = ((n_samples + config.batch_size - 1) // config.batch_size) * config.batch_size
                logging.info(f"\tUsing streaming subset for {split}: {n_samples}/{total_samples} samples")
                dataset[split] = dataset[split].take(n_samples)
            except Exception:
                logging.info(
                    f"Warning: Could not determine split size for streaming subset on {split}. "
                    "Falling back to full streamed split."
                )
    logging.info("\t...downloading_streaming_dataset complete")
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
    # if streaming:
    #     return initialize_streaming_dataloader(config, keep_all_columns=keep_all_columns, splits=splits)
    logging.info("\t...initializing_dataloader complete")
    dataset_dict = dataset_dict.rename_columns({'coordinates': 'data', 'binding_type': 'labels'})
    logging.info("\tPre-encoding labels and packing coordinates into fixed-size tensors in dataset artifacts...")
    requested_workers = max(1, int(config.num_cpus))
    slow_safetensors_cache_filepath = cache_manager.get_safetensor_cache_path(fast_path=False)
    fast_safetensors_filepath_cache = cache_manager.get_safetensor_cache_path(fast_path=True)
    using_fast_cache = cache_manager.using_fast_cache()
    safetensors_cache_folder = fast_safetensors_filepath_cache if using_fast_cache else slow_safetensors_cache_filepath
  
    if using_fast_cache:
        logging.info(f"Copying cached dataset in {slow_safetensors_cache_filepath} for to {fast_safetensors_filepath_cache} directory for faster access during this run...")
        os.makedirs(os.path.dirname(fast_safetensors_filepath_cache), exist_ok=True)
        shutil.copytree(slow_safetensors_cache_filepath, fast_safetensors_filepath_cache, dirs_exist_ok=True)
        cache_manager.add_directory_to_cleanup(fast_safetensors_filepath_cache)
        logging.info(f"\tCopy complete. Using {fast_safetensors_filepath_cache} for {split} split during this run.")
    logging.info(f"\t\tLoading cached safetensors for {split} split from {safetensors_cache_folder}...")
    for split in splits:
        dataset_dict[split] = ShardedSafetensorsDataset(safetensors_cache_folder, split)
    logging.info("\tDataset map preprocessing done; using fast fixed-shape batch path.")

    requested_workers = max(1, int(config.num_cpus))
    dataloader_workers = _effective_worker_count(requested_workers)
    logging.info(
        f"\tUsing {dataloader_workers} DataLoader workers for packed large-tensor batches "
        f"(requested {requested_workers}, hyperthread-aware)."
    )
    dataloader_args = {
        "batch_size": config.batch_size,
        "num_workers": dataloader_workers,
        "persistent_workers": dataloader_workers > 0,
        "pin_memory": True,  # ← Enable for faster CPU→GPU transfer
        "prefetch_factor": 4 if dataloader_workers > 0 else None,
    }
    
    dataloader_dict = {}
    for split in splits:
        logging.info(f"\t\tCreating dataloader for {split} split with batch size {config.batch_size} and num_workers {dataloader_workers}...")
        dataloader = torch.utils.data.DataLoader(dataset_dict[split], **dataloader_args)
        dataloader_dict[split] = dataloader
    cache_manager.copy_to_permanent_cache()
    # print the location of the dataloader on disk for debugging
    for split_name, dataset in dataset_dict.items():
        if hasattr(dataset, 'cache_files'):
            logging.info(dataset.cache_files)    

    return dataloader_dict

# def initialize_streaming_dataloader(config: TrainingConfig, keep_all_columns: bool = False, splits: list = ["train", "validation", "test"], shuffle: bool = False) -> DataLoaderDict:
    """
    Initialize the streaming dataloader for training.
    """
    # Initialize the streaming dataloader
    logging.info("Initializing streaming dataloader...")
    dataset_dict: datasets.IterableDatasetDict = _download_streaming_dataset(config, splits=splits, shuffle=shuffle)

    dataset_dict = dataset_dict.rename_columns({'coordinates': 'data', 'binding_type': 'labels'})
    logging.info("\tEncoding labels and packing coordinates in streaming pipeline...")
    for split in splits:
        if split in dataset_dict:
            batch_size = max(1, int(config.transform_batch_size))
            requested_workers = max(1, int(config.num_cpus))
            num_workers = _effective_worker_count(requested_workers)
            logging.info(
                f"\t\tPreparing streaming map for {split} split with batch size {batch_size} and num_workers {num_workers} "
                f"(requested {requested_workers}, hyperthread-aware)..."
            )
            while True:

                try:

                    dataset_dict[split] = dataset_dict[split].map(
                        _encode_and_pack_batch,
                        batched=True,
                        batch_size=batch_size,
                        num_proc=num_workers,
                        desc=f"Encoding labels and packing coords for {split} (streaming)",
                    )
                    
                    break
                # if out of memory error occurs during map, reduce batch size
                except Exception as e:
                    if _is_host_oom_error(e) and batch_size > 1:
                        logging.info(f"Host-memory OOM during streaming map for {split} split with batch size {batch_size}. Reducing batch size and retrying...")
                        batch_size = max(1, (3*batch_size) // 4)
                        logging.info(f"\tNew batch size: {batch_size}")
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

    for split in splits:
        if split in dataset_dict:
            dataset_dict[split] = _set_torch_format_for_packed_dataset(
                dataset_dict[split],
                keep_all_columns=keep_all_columns,
            )

    dataloader_dict = {}
    for split in splits:
        logging.info(f"\t\tPreparing dataloader for {split} split (on-the-fly preprocessing)...")
        requested_workers = max(1, int(config.num_cpus))
        dataloader_workers = _effective_worker_count(requested_workers)
        logging.info(
            f"\tUsing {dataloader_workers} DataLoader workers for packed large-tensor batches "
            f"(requested {requested_workers}, hyperthread-aware)."
        )
        dataloader_args = {
            "batch_size": config.batch_size,
            "num_workers": dataloader_workers,
            "pin_memory": False,
            "persistent_workers": dataloader_workers > 0,
        }
        if dataloader_workers > 0:
            dataloader_args["prefetch_factor"] = 8
        dataloader = torch.utils.data.DataLoader(dataset_dict[split], **dataloader_args)
        dataloader_dict[split] = dataloader
    logging.info("\t...streaming dataloaders ready")

    resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))
    
    logging.info("\t...initializing_streaming_dataloader complete")
    return dataloader_dict

def clear_cache(remove_dataset_cache: bool = False, remove_mapped_cache: bool = True):
    """
    Clear the Hugging Face dataset cache directory.
    """
    cache_dir = os.environ.get("HF_DOWNLOADED_DATASET_DIR")
    mapped_cache_dir = os.environ.get("HF_MAPPED_DATASET_DIR")

    if remove_dataset_cache and os.path.exists(cache_dir):
        logging.info(f"Clearing dataset cache directory: {cache_dir}")
        shutil.rmtree(cache_dir)
        logging.info("\t...dataset cache cleared")

    if remove_mapped_cache and os.path.exists(mapped_cache_dir):
        logging.info(f"Clearing mapped cache directory: {mapped_cache_dir}")
        shutil.rmtree(mapped_cache_dir)
        logging.info("\t...mapped cache cleared")
    else:
        logging.info(f"Cache directory {cache_dir} does not exist, nothing to clear.")



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
    logging.info("Dataloaders initialized and cached successfully.")
    
    