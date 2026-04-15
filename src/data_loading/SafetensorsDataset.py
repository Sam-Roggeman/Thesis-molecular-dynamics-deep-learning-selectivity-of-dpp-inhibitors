from safetensors import safe_open
import torch

class SafetensorsDataset(torch.utils.data.Dataset):
    """Memory-mapped safetensors dataset for fast loading"""
    
    def __init__(self, filepath):
        self.filepath = filepath
        # Open with zero-copy (no data loaded yet)
        self._tensors = None
    
    def _load(self):
        if self._tensors is None:
            self._tensors = {}
            with safe_open(self.filepath, framework="pt", device="cpu") as f:
                for key in f.keys():
                    self._tensors[key] = f.get_tensor(key)  # Zero-copy!
    
    def __getitem__(self, idx):
        self._load()
        return {
            'data': self._tensors['data'][idx],
            'labels': self._tensors['labels'][idx],
            'num_atoms': self._tensors['num_atoms'][idx],
        }
    
    def __len__(self):
        self._load()
        return len(self._tensors['data'])
    
class ShardedSafetensorsDataset(torch.utils.data.Dataset):
    """Loads sharded safetensors with memory mapping for efficiency."""
    
    def __init__(self, cache_path, split):
        # Load metadata
        metadata = torch.load(f"{cache_path}/{split}_metadata.pt")
        self.total_samples = metadata['total_samples']
        self.n_shards = metadata['n_shards']
        self.cache_path = cache_path
        self.split = split
        
        # Pre-compute sample to shard mapping without loading data
        self.shard_for_sample = []
        for shard_idx in range(self.n_shards):
            shard_path = f"{cache_path}/{split}_shard_{shard_idx:04d}_of_{self.n_shards:04d}.safetensors"
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                shard_size = f.get_tensor("data").shape[0]
                self.shard_for_sample.extend([shard_idx] * shard_size)
    
    def __getitem__(self, idx):
        shard_idx = self.shard_for_sample[idx]
        
        # Lazy load shard (cached for repeated accesses)
        if not hasattr(self, '_cached_shard') or self._cached_shard_idx != shard_idx:
            shard_path = f"{self.cache_path}/{self.split}_shard_{shard_idx:04d}_of_{self.n_shards:04d}.safetensors"
            self._cached_shard = {}
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    self._cached_shard[key] = f.get_tensor(key)
            self._cached_shard_idx = shard_idx
        
        # Find local index within shard
        # For simplicity, this recomputes - you could precompute local indices
        offset = 0
        for i in range(shard_idx):
            prev_shard_path = f"{self.cache_path}/{self.split}_shard_{i:04d}_of_{self.n_shards:04d}.safetensors"
            with safe_open(prev_shard_path, framework="pt", device="cpu") as f:
                offset += f.get_tensor("data").shape[0]
        local_idx = idx - offset
        
        return {
            'data': self._cached_shard['data'][local_idx],
            'labels': self._cached_shard['labels'][local_idx],
            'num_atoms': self._cached_shard['num_atoms'][local_idx],
        }
    
    def __len__(self):
        return self.total_samples