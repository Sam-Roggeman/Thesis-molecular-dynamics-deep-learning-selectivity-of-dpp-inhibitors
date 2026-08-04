import multiprocessing as mp

from src.data_loading.SafetensorsDataset import ShardedSafetensorsDataset
mp.set_start_method('spawn', force=True)
import shutil

import os
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from src.utils.cacheManager import cacheManager
from src.utils.training_config import TrainingConfig
from src.utils.logger import get_logger
logging = get_logger()


def _effective_worker_count(requested_cpus: int) -> int:
    """Use physical-core-like worker count on hyperthreaded systems."""
    requested = max(1, int(requested_cpus))
    return max(1, requested // 2)


# Define a type for the dataloader dict
DataLoaderDict = dict[str, torch.utils.data.DataLoader]

def initialize_dataloaders(config: TrainingConfig, cache_manager: cacheManager= None, splits=None, keep_all_columns: bool = False, streaming: bool = False) -> DataLoaderDict:
    """
    Initialize the dataloader for training.
    """
    if splits is None:
        splits = ['train', 'validation', 'test']
    if cache_manager is None:
        cache_manager = cacheManager()

    logging.info("\t...initializing_dataloader complete")
    logging.info("\tPre-encoding labels and packing coordinates into fixed-size tensors in dataset artifacts...")
    requested_workers = max(1, int(config.num_cpus))
    slow_safetensors_cache_filepath = cache_manager.get_safetensor_cache_path(fast_path=False)
    safetensors_cache_folder = slow_safetensors_cache_filepath

    logging.info(f"\tLoading cached safetensors from {safetensors_cache_folder}...")
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
        logging.debug(f"\t\tLoading cached safetensors for {split} split from {safetensors_cache_folder}...")
        logging.debug(f"Creating SafetensorsDataset for {split} split with batch size {config.batch_size} and num_workers {dataloader_workers}, fraction {config.dataset_size}...")
        ds = ShardedSafetensorsDataset(safetensors_cache_folder, split, fraction=config.dataset_size)
        logging.info(f"\t\tCreating dataloader for {split} split with batch size {config.batch_size} and num_workers {dataloader_workers}...")
        dataloader = torch.utils.data.DataLoader(dataset=ds, **dataloader_args)
        dataloader_dict[split] = dataloader
    cache_manager.copy_to_permanent_cache() 

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



