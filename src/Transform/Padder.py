import torch
import numpy as np

class Padder:
    def __init__(self, target_size=168, fill=0.0):
        self.target_size = target_size
        self.fill = fill

    def __call__(self, data_examples):
        """Pad entries to target_size x target_size efficiently using numpy."""
        target_pixels = self.target_size * self.target_size
        
        # Convert to numpy arrays if needed
        if isinstance(data_examples, list) and len(data_examples) > 0 and isinstance(data_examples[0], list):
            data_examples = [np.array(entry, dtype=np.float32) for entry in data_examples]
        
        # Preallocate padded array for efficiency
        padded_batch = []
        for entry in data_examples:
            if isinstance(entry, np.ndarray):
                n_current = entry.shape[0]
            else:
                entry = np.array(entry, dtype=np.float32)
                n_current = entry.shape[0]
            
            n_pad = target_pixels - n_current
            if n_pad > 0:
                # Use np.pad for efficient padding (much faster than list concatenation)
                padded_entry = np.pad(entry, ((0, n_pad), (0, 0)), mode='constant', constant_values=self.fill)
            else:
                padded_entry = entry[:target_pixels]  # Trim if over size
            padded_batch.append(padded_entry)
        
        return padded_batch

    def reapply_padding(self, examples_data, real_nr_atoms):
        """Reapply padding to a batch of examples based on their real number of atoms"""
        padded_data = []
        for i, data in enumerate(examples_data):
            n_real_atoms = real_nr_atoms[i]
            n_total_pixels = self.target_size * self.target_size
            n_pad_pixels = n_total_pixels - n_real_atoms
            padded_example = np.vstack([data[:n_real_atoms], np.full((n_pad_pixels, 3), self.fill, dtype=np.float32)])
            padded_data.append(padded_example)
        return padded_data
