from dotenv import load_dotenv
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait

from src.infrastructure.cacheManager import cacheManager
load_dotenv()
from src.training.batch_preprocessing import TARGET_PIXELS

from src.training.LabelEncoder import LabelEncoder
from src.data.data_loading.SafetensorsDataset import ShardedSafetensorsDataset
import datasets
# The label encoder is shared by the batch packing step so that string labels are
# converted to stable numeric IDs before being written to disk.
label_encoder = LabelEncoder()
import logging
logging.basicConfig(level=logging.DEBUG)
import os 
from src.config.training_config import calculate_num_cpus
import numpy as np
from safetensors.torch import save_file, safe_open
import torch
import shutil
import time
import signal
import threading
import sys

# A global event is used so the SIGUSR1 handler can request a graceful shutdown
# from anywhere in the preprocessing pipeline without raising immediately in the
# signal handler itself.
stop_requested = threading.Event()


class GracefulStopRequested(SystemExit):
    """Raised when GPULab requests a graceful shutdown (SIGUSR1)."""


def handle_sigusr1(signum, frame):
    # The handler only flips the shutdown flag; the actual exit is raised later
    # from regular Python code at safe checkpoints.
    logging.warning("Received SIGUSR1, requesting graceful stop...")
    stop_requested.set()


def _raise_if_stop_requested(context: str = ""):
    # Centralized cancellation check so long-running loops can stop cleanly and
    # report where the request was observed.
    if stop_requested.is_set():
        suffix = f" during {context}" if context else ""
        raise GracefulStopRequested(123, f"Graceful stop requested{suffix}")


signal.signal(signal.SIGUSR1, handle_sigusr1)

# When set, worker subprocesses can slice the current split without serializing
# the entire dataset into every child process.
_WORKER_DATASET_SPLIT = None

def _write_safetensor_shard_from_worker_split(shard_idx, n_shards, start, end, shard_path, num_workers=1):
    """Materialize and write one shard inside a subprocess worker.

    The parent process passes a slice of the dataset for this shard. This helper
    splits that slice into smaller chunks, materializes each chunk in parallel,
    concatenates the arrays back together, and then writes the shard to disk
    atomically so partially written files are never treated as valid output.
    """
    _raise_if_stop_requested(f"shard {shard_idx + 1}/{n_shards} setup")

    # Break the shard range into smaller pieces so worker processes can materialize
    # independent slices without holding the entire shard in memory at once.
    chunk_size = max(1, (end - start) // num_workers)
    chunks = []
    for i in range(0, end - start, chunk_size):
        chunk_end = min(i + chunk_size, end - start)
        chunks.append((start+i, start+chunk_end))

    # This log message is useful when a shard stalls, because it shows both the
    # overall sample range and the degree of internal parallelism being used.
    logging.info(f"Worker for shard {shard_idx+1}/{n_shards} processing samples {start}-{end} with {num_workers} workers and chunk size {chunk_size}...")

    # Each chunk is sent to a separate process so the expensive dataset slicing
    # and numpy conversion can overlap across CPU cores.
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for chunk_start, chunk_end in chunks:
            _raise_if_stop_requested(f"shard {shard_idx + 1}/{n_shards} chunk scheduling")
            future = executor.submit(
                # Use the module-level split to avoid copying the full dataset slice
                # into every process submission.
                _materialize_shard_absolute,
                chunk_start,
                chunk_end
            )
            futures.append(future)

        # Collect results while periodically checking for shutdown requests so
        # SIGUSR1 can stop the pipeline without waiting for the entire pool to finish.
        chunk_results = []
        pending = set(futures)
        while pending:
            _raise_if_stop_requested(f"shard {shard_idx + 1}/{n_shards} chunk execution")
            done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
            for completed in done:
                # Any exception raised inside a worker should surface here so the
                # caller sees the actual failure instead of silently continuing.
                chunk_results.append(completed.result())

        # One final check before assembling the arrays. This keeps cancellation
        # responsive even if the pool finished successfully but a shutdown was
        # requested while results were being gathered.
        _raise_if_stop_requested(f"shard {shard_idx + 1}/{n_shards} post-processing")
    
    # Reassemble the per-chunk outputs into one contiguous block per field so the
    # resulting tensors can be written as a single safetensors shard.
    data_np = np.concatenate([r[0] for r in chunk_results], axis=0)
    labels_np = np.concatenate([r[1] for r in chunk_results], axis=0)
    num_atoms_np = np.concatenate([r[2] for r in chunk_results], axis=0)

    # Convert the numpy arrays into torch tensors for safetensors, forcing a
    # contiguous layout for the large coordinate tensor before serialization.
    tensors = {
        'data': torch.from_numpy(np.ascontiguousarray(data_np)),
        'labels': torch.from_numpy(labels_np),
        'num_atoms': torch.from_numpy(num_atoms_np),
    }

    # Write to a temporary path first, then replace the final shard path only once
    # the file is complete. This avoids leaving behind truncated shard files.
    temp_path = f"{shard_path}.part"
    try:
        # Remove any stale temp file from a previous interrupted run so the new
        # write starts from a clean slate.
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

        # Serialize the shard to the temporary location.
        save_file(tensors, temp_path)

        # Fsync the file so the payload is flushed to disk before the rename.
        try:
            with open(temp_path, 'rb') as f:
                os.fsync(f.fileno())
        except Exception:
            logging.debug(f"Could not fsync temp shard file {temp_path}")

        # Move the shard into its final path atomically. If this succeeds, readers
        # will either see the old valid file or the new valid file, never a half-written one.
        os.replace(temp_path, shard_path)

        # Fsync the directory entry as well so the rename itself is durable on disk.
        try:
            dirfd = os.open(os.path.dirname(shard_path) or '.', os.O_DIRECTORY)
            try:
                os.fsync(dirfd)
            finally:
                os.close(dirfd)
        except Exception:
            logging.debug(f"Could not fsync directory for {shard_path}")

        return shard_idx, n_shards, shard_path, os.path.getsize(shard_path)
    finally:
        # Clean up stray temp files if the write failed midway through.
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass


def _materialize_shard_absolute(start, end):
    """Read one shard slice once, then extract arrays for all required fields."""
    logging.debug(f"Worker for {start}-{end} starting materialization...")
    _raise_if_stop_requested(f"materialization {start}-{end}")
    start_time = time.time()
    global _WORKER_DATASET_SPLIT
    # Slice the already-loaded split once, then convert each column into numpy
    # arrays that can be concatenated and serialized efficiently.
    shard_batch = _WORKER_DATASET_SPLIT[start:end]
    data_np = np.ascontiguousarray(np.asarray(shard_batch['data']))
    labels_np = np.asarray(shard_batch['labels'])
    num_atoms_np = np.asarray(shard_batch['num_atoms'])
    end_time = time.time()
    logging.debug(f"Worker for {start}-{end} materialized batch in {end_time - start_time:.2f} seconds")
    return data_np, labels_np, num_atoms_np


def _materialize_shard_arrays(dataset_split, start, end):
    """Read one shard slice once, then extract arrays for all required fields."""
    # This helper performs the same conversion as the worker version, but takes
    # the dataset slice explicitly instead of using the global worker cache.
    shard_batch = dataset_split[start:end]
    data_np = np.ascontiguousarray(np.asarray(shard_batch['data']))
    labels_np = np.asarray(shard_batch['labels'])
    num_atoms_np = np.asarray(shard_batch['num_atoms'])
    return data_np, labels_np, num_atoms_np


def _download_dataset(dataset_path, splits=None) -> datasets.DatasetDict:
    """Download the dataset using Hugging Face's `datasets` library."""
    # All dataset loading and caching behavior is centralized here so the rest of
    # the pipeline can assume it receives a ready-to-use DatasetDict.
    logging.info("Downloading dataset from Hugging Face...")
    cache_dir = os.environ.get("HF_DOWNLOADED_DATASET_DIR")
    if not cache_dir:
        logging.warning("HF_DOWNLOADED_DATASET_DIR not set. Using default cache directory.")
    logging.info(f"Only downloading splits: {splits}")
    _split_arg = {}
    for split in splits:
        # Hugging Face expects a split mapping when downloading multiple named
        # splits in one call.
        logging.info(f"\t\tDownloading of {split} split")
        _split_arg[split] = split
    logging.info(f"Downloading dataset from {dataset_path} with splits {_split_arg}...")
    logging.info(f"\tUsing cache directory: {cache_dir}")
    # Use half the CPUs for download/loading work so the machine still has room
    # for the later preprocessing stages and the operating system.
    num_proc=calculate_num_cpus()//2
    logging.info(f"\tUsing {num_proc} CPU workers for dataset loading")

    logging.debug(f"Dataset loading parameters: dataset_location={dataset_path}, splits={splits}, cache_dir={cache_dir}, num_proc={num_proc}")
    dataset_dict: datasets.DatasetDict = datasets.load_dataset(
        dataset_path,
        split=_split_arg,
        cache_dir=cache_dir,
        token=os.environ.get("HF_TOKEN"),
        num_proc=num_proc,
    )
    logging.info("\t...downloading_dataset complete")
    # Older call paths may return a single Dataset instead of a DatasetDict; wrap
    # it so the rest of the code can treat every result uniformly.
    if not isinstance(dataset_dict, datasets.DatasetDict):
        dataset_dict = datasets.DatasetDict({splits[0]: dataset_dict})
    return dataset_dict

def calculate_sample_size(sample, prefix="sample"):
    """Recursively estimate sample size in bytes and print per-field breakdown."""
    # Arrays are the primary payload type written to safetensors.
    if isinstance(sample, np.ndarray):
        size = sample.nbytes
        return size

    # Torch tensors are handled explicitly so the size estimate remains accurate
    # if a sample has already been converted from numpy.
    if isinstance(sample, torch.Tensor):
        size = sample.element_size() * sample.nelement()
        return size

    # Nested dictionaries are walked recursively so structured samples can still
    # be measured field by field.
    if isinstance(sample, dict):
        total_size = 0
        for key, value in sample.items():
            total_size += calculate_sample_size(value, prefix=f"{prefix}.{key}")
        return total_size

    # Lists and tuples are also traversed recursively because some datasets store
    # coordinates or metadata in sequence containers before packing.
    if isinstance(sample, (list, tuple)):
        total_size = 0
        for idx, value in enumerate(sample):
            total_size += calculate_sample_size(value, prefix=f"{prefix}[{idx}]")
        return total_size

    # Fallback: estimate scalar/object size from its UTF-8 representation.
    size = len(str(sample).encode("utf-8"))  # Rough estimate for scalar/object data.
    return size

def _save_split_as_safetensors_memory_efficient(dataset_dict, split, cache_path, shard_size=5000, total_nr_workers=1):
    """
    Memory-efficient saving with sharding.
    
    shard_size: Adjust based on available RAM.
    - Each shard uses ~shard_size * 168 * 3 * 4 bytes for data
    - Example: 5000 samples * 168 * 168 * 3 * 32 bits / 8 bits/byte  = 1.69344 GB per shard
    """
    logging.info(f"Saving {split} split to safetensors with shard size {shard_size} and {total_nr_workers} workers to cache path {cache_path}...")
    # Determine how many samples must be written for this split.
    total_samples = len(dataset_dict[split])
    logging.info(f"Saving {total_samples} samples for {split} split")
    
    # Convert the requested shard size into a shard count rounded up to cover
    # the tail of the split.
    n_shards = (total_samples + shard_size - 1) // shard_size
    logging.info(f"Creating {n_shards} shards of ~{shard_size} samples each")
    # Sample one row to estimate the eventual shard size and log a sanity check.
    sample = dataset_dict[split][0] # Get the first sample to estimate size
    total_size = calculate_sample_size(sample)

    logging.info(f"Size per sample: {total_size / 1024:.2f} KB")
    logging.info(f"Estimated size per shard: {(total_size * shard_size) / 1024 / 1024/1024:.2f} GB")

    logging.info(
        f"Saving shards with {total_nr_workers} processes. "
        "Each worker materializes and writes its own shard, so tune workers conservatively."
    )

    logging.debug(f"Total available workers: {total_nr_workers}") 

    _raise_if_stop_requested(f"split {split} pre-save")
    # Expose the current split to worker subprocesses so they can materialize
    # chunks directly without receiving a serialized copy of the whole split.
    global _WORKER_DATASET_SPLIT
    _WORKER_DATASET_SPLIT = dataset_dict[split]
    for shard_idx in range(n_shards):
        _raise_if_stop_requested(f"split {split} shard loop")
        start = shard_idx * shard_size
        end = min((shard_idx + 1) * shard_size, total_samples)
        # This log makes it easy to identify which sample window is being written
        # if one shard fails or takes much longer than the others.
        logging.info(f"Processing shard {shard_idx+1}/{n_shards} (samples {start}-{end}) for saving)")

        shard_path = f"{cache_path}/{split}_shard_{shard_idx:04d}_of_{n_shards:04d}.safetensors"
        if os.path.exists(shard_path):
            # Verify integrity before skipping: an incomplete file can cause deserialization errors.
            try:
                # Opening the shard confirms that the file is at least readable as a
                # safetensors archive before we skip rewriting it.
                with safe_open(shard_path, framework="pt", device="cpu") as _:
                    logging.info(f"  Shard {shard_idx+1} already exists at {shard_path}, valid. Skipping...")
                    continue
            except Exception as e:
                logging.warning(f"  Existing shard {shard_path} is corrupted or incomplete: {e}. Removing and rewriting.")
                try:
                    os.remove(shard_path)
                except Exception:
                    logging.warning(f"  Failed to remove corrupted shard {shard_path}; will attempt to overwrite.")
        _write_safetensor_shard_from_worker_split(
            shard_idx,
            n_shards,
            start,
            end,
            shard_path,
            num_workers=total_nr_workers,
        )

    # Clear the global reference once saving is complete so the worker state does
    # not accidentally linger for later operations in the same process.
    _WORKER_DATASET_SPLIT = None
        
    
    # Store a tiny metadata file next to the shards so downstream code can inspect
    # the split without scanning every safetensors file.
    metadata = {
        'total_samples': total_samples,
        'n_shards': n_shards,
        'shard_size': shard_size,
        'split': split
    }
    torch.save(metadata, f"{cache_path}/{split}_metadata.pt")
    
    return ShardedSafetensorsDataset(cache_path, split)
def _encode_and_pack_batch(batch):
    # The Hugging Face map step feeds batches of rows into this function. It is
    # responsible for making the dataset ready for safetensors serialization.
    labels = batch.get("labels")
    data = batch.get("data")

    # If neither field is present, leave the batch unchanged so the pipeline can
    # safely pass through unexpected records.
    if labels is None and data is None:
        return batch

    encoded = []
    if labels is not None:
        # String labels are encoded to integers so they can be stored efficiently
        # and used by downstream training code without extra preprocessing.
        for label in labels:
            if isinstance(label, str):
                encoded.append(label_encoder.encode_label(label))
            else:
                encoded.append(int(label))

    if data is None:
        # Some batches may only contain labels. In that case, return the encoded
        # labels without attempting to pack coordinate data.
        return {"labels": encoded}

    # Pack variable-length coordinate arrays into a fixed-size tensor so each row
    # has the same shape and can be written to a dense safetensors shard.
    batch_size = len(data)
    packed = np.zeros((batch_size, TARGET_PIXELS, 3), dtype=np.float32)
    num_atoms = np.zeros((batch_size,), dtype=np.int16)
    for i, coords in enumerate(data):
        arr = np.asarray(coords, dtype=np.float32)
        # Accept both flattened and 2D coordinate representations and normalize
        # them into an ``(n_atoms, 3)`` array.
        if arr.ndim == 1:
            arr = arr.reshape(-1, 3)
        elif arr.ndim != 2 or arr.shape[1] != 3:
            arr = arr.reshape(-1, 3)

        # Truncate or pad each sample to the fixed target length expected by the
        # model training code.
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
    # MemoryError is the most direct indicator of a host-side allocation failure.
    if isinstance(exc, MemoryError):
        return True

    # These string markers catch the most common failure modes emitted by Python,
    # PyTorch, and underlying allocator libraries.
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
        # Walk the causal chain so wrapper exceptions still count as OOMs when the
        # actual failure is nested deeper in ``__cause__`` or ``__context__``.
        if isinstance(current, MemoryError):
            return True
        msg = f"{type(current).__name__}: {current}".lower()
        if any(marker in msg for marker in oom_markers):
            return True
        current = current.__cause__ or current.__context__

    return False


def _is_map_worker_crash_error(exc: BaseException) -> bool:
    """Return True when Hugging Face map multiprocessing workers crash."""
    # These messages occur when the underlying process pool dies rather than
    # failing with a clean Python exception.
    markers = (
        "one of the subprocesses has abruptly died during map operation",
        "a worker process managed by the executor was unexpectedly terminated",
        "brokenprocesspool",
        "a process in the process pool was terminated abruptly while the future was running or pending",
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

def prepare_safetensors(splits=["train", "validation", "test", "unique_test_runs"], initial_batch_size=1024, skip_existing_cache=True,dataset_path="/project_antwerp/dataset/temp/streaming_pdb_dataset/", redo_cache=False):
    # This is the top-level orchestration routine: download the requested splits,
    # normalize them into fixed-size tensors, and persist the result as sharded
    # safetensors on disk.
    cache_prefix = os.environ.get("HF_DOWNLOADED_DATASET_DIR", "./hf_cache")
    dataset_dict = _download_dataset(dataset_location=dataset_path, splits=splits)
    logging.debug(f"Downloaded dataset with splits: {list(dataset_dict.keys())}. Sample keys: {dataset_dict[splits[0]].column_names}")
    
    # Start with a relatively large batch size; if the environment cannot handle
    # it, the retry loop below reduces the batch size automatically.
    batch_size = initial_batch_size  # Start with a larger batch size for the map operation.
    # Use the configured CPU count for parallel packing/saving work.
    num_workers = calculate_num_cpus() 
    cache_manager = cacheManager.cacheManager()
    # Build a stable cache namespace so repeated runs can reuse or invalidate the
    # same disk locations in a predictable way.
    dir_name = cacheManager.construct_cache_identifier(used_percentage="100%", prefix="mapped")
    safetensors_cache_filepath = cache_manager.get_safetensor_cache_path(fast_path=False)
    arrow_cache_filepath = os.path.join(cache_prefix, dir_name, "arrow_cache")
    # Rename columns to the schema expected by the packing and safetensors-saving
    # steps. ``coordinates`` becomes ``data`` and ``binding_type`` becomes labels.
    dataset_dict = dataset_dict.rename_columns({'coordinates': 'data', 'binding_type': 'labels'})
    if redo_cache and os.path.exists(safetensors_cache_filepath):
        # A forced rebuild starts by deleting the old cache directory entirely.
        logging.warning(f"\t\tRedo cache enabled. Will overwrite existing safetensors cache for {safetensors_cache_filepath}.")
        shutil.rmtree(safetensors_cache_filepath)
    os.makedirs(safetensors_cache_filepath, exist_ok=True)
    
    for split in splits:
        _raise_if_stop_requested(f"split {split} start")
        cache_metadata_path = os.path.join(safetensors_cache_filepath, f"{split}_metadata.pt")
        logging.debug(f"Checking for existing cache metadata at {cache_metadata_path} for {split} split...")
        if skip_existing_cache and os.path.exists(cache_metadata_path) and not redo_cache:
            # Metadata presence is used as the completion signal for a cached split.
            logging.info(f"\t\tFound existing cached safetensors for {split} split in {safetensors_cache_filepath}.")
            continue  # Skip if already cached (metadata presence indicates completed cache)

        arrow_cache_path = os.path.join(arrow_cache_filepath, f"{split}_data.arrow")
        # Map first, save second: this keeps the intermediate Arrow cache separate
        # from the final safetensors shards and makes retry logic easier.
        logging.info(f"\t\tProcessing {split} split with batch size {batch_size} and {num_workers} workers to cache {arrow_cache_path}...")
        if redo_cache and os.path.exists(arrow_cache_filepath):
            shutil.rmtree(arrow_cache_filepath)
            logging.warning(f"\t\tRedo cache enabled. Removed existing arrow cache at {arrow_cache_filepath} for {split} split.")

        while True:
            _raise_if_stop_requested(f"split {split} map")
            try: 
                # The map step normalizes raw dataset rows into fixed-size tensors and
                # encoded labels, which are then cached in Arrow format.
                dataset_dict[split] = dataset_dict[split].map(
                    _encode_and_pack_batch,
                    batched=True,
                    batch_size=batch_size,
                    num_proc=num_workers,
                    desc=f"Encoding labels and packing coords for {split}",
                    cache_file_name=arrow_cache_path,
                )
                logging.info(f"\t\tEncoding and packing complete for {split} split. Saving to safetensors cache...")
                break
            except Exception as e:
                logging.error(f"Error during map for {split} split with batch size {batch_size}: {e}")
                # If a host-memory OOM occurs during map, reduce batch size and retry.
                if( _is_host_oom_error(e) or _is_map_worker_crash_error(e) ) and batch_size > 1:
                    # Back off conservatively so the next attempt uses less memory per
                    # map call and has a better chance of surviving.
                    logging.info(f"Host-memory OOM during map for {split} split with batch size {batch_size}. Reducing batch size and retrying...")
                    batch_size = max(1, (3*batch_size) // 4)
                    logging.info(f"\tNew batch size: {batch_size}")
                    continue
                else:
                    raise 
        while True:
            _raise_if_stop_requested(f"split {split} safetensor save")
            try:
                # Convert the Arrow-cached split into sharded safetensors using the
                # same worker-aware path above.
                _save_split_as_safetensors_memory_efficient(
                    dataset_dict, 
                    split, 
                    cache_path=safetensors_cache_filepath,
                    shard_size=5000,  # Adjust based on your memory
                    total_nr_workers=num_workers,
                )
                break
            except Exception as e:
                logging.error(f"Error during saving safetensors for {split} split: {e}")
                if (_is_host_oom_error(e) or _is_map_worker_crash_error(e)) and num_workers > 1:
                    # If saving fails because the process pool is too aggressive, back
                    # off by halving the worker count and try again.
                    logging.info(
                        f"Worker failure during safetensors saving for {split} with {num_workers} workers. "
                        "Reducing workers and retrying..."
                    )
                    num_workers = max(1, num_workers // 2)
                    logging.info(f"\tNew number of workers: {num_workers}")
                    continue
                else:
                    raise
        logging.info(f"\t\tFinished processing {split} split. Cached safetensors available at {safetensors_cache_filepath}.")
    logging.info("All splits processed and cached as safetensors.")


if __name__ == "__main__":
    dataset_path = "/project_antwerp/dataset/temp/streaming_pdb_dataset/"
    try:
        # Allow the module to be executed directly for local preprocessing runs
        # without needing a separate wrapper script.
        prepare_safetensors(
            splits=["train", "validation", "test", "unique_test_runs"],
            initial_batch_size=2048,
            skip_existing_cache=False,
            dataset_path=dataset_path,
            redo_cache=False,
        )
    except GracefulStopRequested as exc:
        exit_code = exc.code if isinstance(exc.code, int) else 123
        logging.warning(f"Graceful stop acknowledged. Exiting with code {exit_code}.")
        sys.exit(exit_code)
    