"""Lazy PyTorch datasets backed by single or sharded safetensors files."""

from safetensors import safe_open
import torch

class SafetensorsDataset(torch.utils.data.Dataset):
    """Load ``data``, ``labels``, and ``num_atoms`` from one safetensors file.

    The file is opened lazily on the first length or item request and retained in
    memory for subsequent accesses.
    """
    
    def __init__(self, filepath):
        """Store the file path without loading tensor data."""
        self.filepath = filepath
        # Open with zero-copy (no data loaded yet)
        self._tensors = None
    
    def _load(self):
        """Load all tensors once when the dataset is first accessed."""
        if self._tensors is None:
            self._tensors = {}
            with safe_open(self.filepath, framework="pt", device="cpu") as f:
                for key in f.keys():
                    self._tensors[key] = f.get_tensor(key)  # Zero-copy!
    
    def __getitem__(self, idx):
        """Return one sample with coordinates, label, and valid-atom count."""
        self._load()
        return {
            'data': self._tensors['data'][idx],
            'labels': self._tensors['labels'][idx],
            'num_atoms': self._tensors['num_atoms'][idx],
        }
    
    def __len__(self):
        """Return the number of samples in the backing file."""
        self._load()
        return len(self._tensors['data'])
    
class ShardedSafetensorsDataset(torch.utils.data.Dataset):
    """Read a fraction of sharded safetensors with one-shard caching.

    Each split requires ``{split}_metadata.pt`` and shard files containing the
    keys ``data``, ``labels``, and ``num_atoms``. Only the shard needed for the
    current item is loaded, and consecutive accesses reuse that shard.
    """
    
    def __init__(self, cache_path, split, fraction=1.0):
        """Index the requested split and cap its length to ``fraction``."""
        if not 0 < fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")

        # Load metadata
        metadata = torch.load(f"{cache_path}/{split}_metadata.pt")
        total_samples = metadata['total_samples']
        self.total_samples = min(total_samples, max(1, int(total_samples * fraction)))
        self.n_shards = metadata['n_shards']
        self.cache_path = cache_path
        self.split = split
        
        # Pre-compute sample to shard mapping without loading data.
        # Also keep shard start indices so __getitem__ can compute local_idx in O(1).
        self.shard_for_sample = []
        self.shard_start_idx = {}
        for shard_idx in range(self.n_shards):
            if len(self.shard_for_sample) >= self.total_samples:
                break

            shard_path = f"{cache_path}/{split}_shard_{shard_idx:04d}_of_{self.n_shards:04d}.safetensors"
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                shard_size = f.get_tensor("data").shape[0]

            remaining = self.total_samples - len(self.shard_for_sample)
            take = min(shard_size, remaining)
            if take > 0:
                self.shard_start_idx[shard_idx] = len(self.shard_for_sample)
                self.shard_for_sample.extend([shard_idx] * take)

        self.total_samples = len(self.shard_for_sample)
        self._cached_shard = None
        self._cached_shard_idx = None
    
    def __getitem__(self, idx):
        """Return one sample, loading and caching its source shard if needed."""
        if idx < 0 or idx >= self.total_samples:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.total_samples}")

        shard_idx = self.shard_for_sample[idx]
        
        # Lazy load shard (cached for repeated accesses)
        if self._cached_shard is None or self._cached_shard_idx != shard_idx:
            shard_path = f"{self.cache_path}/{self.split}_shard_{shard_idx:04d}_of_{self.n_shards:04d}.safetensors"
            self._cached_shard = {}
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    self._cached_shard[key] = f.get_tensor(key)
            self._cached_shard_idx = shard_idx
        
        local_idx = idx - self.shard_start_idx[shard_idx]
        
        return {
            'data': self._cached_shard['data'][local_idx],
            'labels': self._cached_shard['labels'][local_idx],
            'num_atoms': self._cached_shard['num_atoms'][local_idx],
        }
    
    def __len__(self):
        """Return the number of samples selected from the shard set."""
        return self.total_samples