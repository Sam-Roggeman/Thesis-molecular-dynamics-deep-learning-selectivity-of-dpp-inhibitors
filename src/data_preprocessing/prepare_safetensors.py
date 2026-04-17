from dotenv import load_dotenv

from src.utils import cacheManager
load_dotenv()
from src.model_training.batch_preprocessing import TARGET_PIXELS

from src.model_training.LabelEncoder import LabelEncoder
from src.data_loading.SafetensorsDataset import ShardedSafetensorsDataset
import datasets
label_encoder = LabelEncoder()
from src.utils.logger import get_logger
logging = get_logger()
import os 
from src.utils.training_config import calculate_num_cpus
import numpy as np
from safetensors.torch import save_file
import torch
def _download_dataset(dataset_location, splits=None) -> datasets.DatasetDict:
    """Download the dataset using Hugging Face's `datasets` library."""
    logging.info("Downloading dataset from Hugging Face...")
    cache_dir = os.environ.get("HF_DOWNLOADED_DATASET_DIR")
    if not cache_dir:
        logging.warning("HF_DOWNLOADED_DATASET_DIR not set. Using default cache directory.")
    logging.info(f"Only downloading splits: {splits}")
    for split in splits:
        logging.info(f"\t\tDownloading of {split} split")
        _split_arg = {
            "train": f"train",
            "validation": f"validation",
            "test": f"test",
        }    
    # cache dir
    logging.info(f"\tUsing cache directory: {cache_dir}")
    num_proc=calculate_num_cpus()//2
    logging.info(f"\tUsing {num_proc} CPU workers for dataset loading")


    dataset_dict: datasets.DatasetDict = datasets.load_dataset(
        dataset_location,
        split=_split_arg,
        cache_dir=cache_dir,
        token=os.environ.get("HF_TOKEN"),
        num_proc=num_proc,
    )
    logging.info("\t...downloading_dataset complete")
    # wrap the dataset in a DatasetDict if it's not already one
    if not isinstance(dataset_dict, datasets.DatasetDict):
        dataset_dict = datasets.DatasetDict({splits[0]: dataset_dict})
    return dataset_dict

def calculate_sample_size(sample, prefix="sample"):
    """Recursively estimate sample size in bytes and print per-field breakdown."""
    if isinstance(sample, np.ndarray):
        size = sample.nbytes
        return size

    if isinstance(sample, torch.Tensor):
        size = sample.element_size() * sample.nelement()
        return size

    if isinstance(sample, dict):
        total_size = 0
        for key, value in sample.items():
            total_size += calculate_sample_size(value, prefix=f"{prefix}.{key}")
        return total_size

    if isinstance(sample, (list, tuple)):
        total_size = 0
        for idx, value in enumerate(sample):
            total_size += calculate_sample_size(value, prefix=f"{prefix}[{idx}]")
        return total_size

    size = len(str(sample).encode("utf-8"))  # Rough estimate for scalar/object data.
    return size

def _save_split_as_safetensors_memory_efficient(dataset_dict, split, cache_path, shard_size=5000):
    """
    Memory-efficient saving with sharding.
    
    shard_size: Adjust based on available RAM.
    - Each shard uses ~shard_size * 168 * 3 * 4 bytes for data
    - Example: 5000 samples * 168 * 168 * 3 * 32 bits / 8 bits/byte  = 1.69344 GB per shard
    """
    
    # Get total size
    total_samples = len(dataset_dict[split])
    logging.info(f"Saving {total_samples} samples for {split} split")
    
    # Create shards
    n_shards = (total_samples + shard_size - 1) // shard_size
    logging.info(f"Creating {n_shards} shards of ~{shard_size} samples each")
    # calculate the size of one sample for debugging
    sample = dataset_dict[split][0] # Get the first sample to estimate size
    total_size = calculate_sample_size(sample)

    logging.info(f"Size per sample: {total_size / 1024:.2f} KB")
    logging.info(f"Estimated size per shard: {(total_size * shard_size) / 1024 / 1024/1024:.2f} GB")

    for shard_idx in range(n_shards):
        start = shard_idx * shard_size
        end = min((shard_idx + 1) * shard_size, total_samples)
        logging.info(f"Processing shard {shard_idx+1}/{n_shards} (samples {start}-{end})")
        
        # Convert to tensors
        tensors = {
            'data': torch.from_numpy(np.array(dataset_dict[split]['data'][start:end])),
            'labels': torch.from_numpy(np.array(dataset_dict[split]['labels'][start:end])),
            'num_atoms': torch.from_numpy(np.array(dataset_dict[split]['num_atoms'][start:end])),
        }
        
        # Save shard
        shard_path = f"{cache_path}/{split}_shard_{shard_idx:04d}_of_{n_shards:04d}.safetensors"
        save_file(tensors, shard_path)
        logging.info(f"  Saved to {shard_path} ({os.path.getsize(shard_path) / 1024 / 1024 / 1024:.1f} GB)")

        # Free memory
        del tensors
        
    
    # Save metadata file for easy loading
    metadata = {
        'total_samples': total_samples,
        'n_shards': n_shards,
        'shard_size': shard_size,
        'split': split
    }
    torch.save(metadata, f"{cache_path}/{split}_metadata.pt")
    
    return ShardedSafetensorsDataset(cache_path, split)
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


def prepare_safetensors(splits=["train", "validation", "test", "unseen_trajects"], initial_batch_size=1024, skip_existing_cache=True,repo_id="Sam-Roggeman/SamRoggeman_Thesis_Dataset_full"):
    cache_prefix = os.environ.get("HF_DOWNLOADED_DATASET_DIR", "./hf_cache")
    dataset_dict = _download_dataset(dataset_location=repo_id, splits=splits)
    batch_size = initial_batch_size  # Start with a larger batch size for the map operation.
    num_workers = calculate_num_cpus() // 2
    cache_manager = cacheManager.cacheManager()
    dir_name = cacheManager.construct_cache_identifier(used_percentage="100%", prefix="mapped")
    safetensors_cache_filepath = cache_manager.get_safetensor_cache_path(fast_path=False)
    arrow_cache_filepath = os.path.join(cache_prefix, dir_name, "arrow_cache")
    dataset_dict = dataset_dict.rename_columns({'coordinates': 'data', 'binding_type': 'labels'})

    os.makedirs(safetensors_cache_filepath, exist_ok=True)
    for split in splits:
        cache_metadata_path = os.path.join(safetensors_cache_filepath, f"{split}_metadata.pt")
        logging.debug(f"Checking for existing cache metadata at {cache_metadata_path} for {split} split...")
        if skip_existing_cache and os.path.exists(cache_metadata_path):
            logging.info(f"\t\tFound existing cached safetensors for {split} split in {safetensors_cache_filepath}.")
            continue  # Skip if already cached (metadata presence indicates completed cache)

        while True:
            try: 
                logging.info(f"\t\tProcessing {split} split with batch size {batch_size} and {num_workers} workers...")
                dataset_dict[split] = dataset_dict[split].map(
                    _encode_and_pack_batch,
                    batched=True,
                    batch_size=batch_size,
                    num_proc=num_workers,
                    desc=f"Encoding labels and packing coords for {split}",
                    cache_file_name=os.path.join(arrow_cache_filepath, f"{split}_data.arrow"),
                )
                logging.info(f"\t\tEncoding and packing complete for {split} split. Saving to safetensors cache...")
                _save_split_as_safetensors_memory_efficient(
                    dataset_dict, 
                    split, 
                    cache_path=safetensors_cache_filepath,
                    shard_size=5000  # Adjust based on your memory
                )
                break
            except Exception as e:
                logging.error(f"Error during map for {split} split with batch size {batch_size}: {e}")
                # If a host-memory OOM occurs during map, reduce batch size and retry.
                if( _is_host_oom_error(e) or _is_map_worker_crash_error(e) ) and batch_size > 1:
                    logging.info(f"Host-memory OOM during map for {split} split with batch size {batch_size}. Reducing batch size and retrying...")
                    batch_size = max(1, (3*batch_size) // 4)
                    logging.info(f"\tNew batch size: {batch_size}")
                    continue
                else:
                    raise
        logging.info(f"\t\tFinished processing {split} split. Cached safetensors available at {safetensors_cache_filepath}.")

if __name__ == "__main__":
    repo_id = "Sam-Roggeman/SamRoggeman_Thesis_Dataset_full"
    prepare_safetensors(splits=["train", "validation", "test", "unseen_trajects"], initial_batch_size=2048, skip_existing_cache=True, repo_id=repo_id)
    