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