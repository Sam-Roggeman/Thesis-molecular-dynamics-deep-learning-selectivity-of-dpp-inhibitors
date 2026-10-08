"""Convert padded XYZ coordinate arrays into channel-first RGB tensors."""

import torch
import numpy as np

class XYZToRGBTensor:
    """Normalize XYZ coordinates into ``[C, H, W]`` image representations.

    Only real atoms participate in min/max normalization; padded atoms become
    black. A list input returns a list of images, while an array input returns a
    stacked NumPy array.
    """

    def __init__(self, target_size=168):
        """Set the square image edge length used for reshaping coordinates."""
        self.target_size = target_size

    def __call__(self, coords_batch, num_real_atoms_batch=None):
        """
        Args:
            coords_batch: List or array of padded ``[N, 3]`` coordinate arrays.
            num_real_atoms_batch: Optional count of real atoms per sample.
        Returns:
            List or array of float32 images with shape ``[3, target_size, target_size]``.
        """
        # Efficient handling: keep as list to avoid unnecessary array copies
        if isinstance(coords_batch, list):
            is_list = True
            batch_size = len(coords_batch)
        else:
            is_list = False
            batch_size = coords_batch.shape[0]

        # Process each sample individually to avoid large intermediate arrays
        img_batch_list = []
        
        for i in range(batch_size):
            if is_list:
                coords = coords_batch[i] if isinstance(coords_batch[i], np.ndarray) else np.array(coords_batch[i], dtype=np.float32)
            else:
                coords = coords_batch[i]
            
            # Get the number of real atoms for this sample
            num_real = num_real_atoms_batch[i] if num_real_atoms_batch is not None else coords.shape[0]

            # Compute min/max only from real atoms (excluding padding)
            real_coords = coords[:num_real]
            coords_min = real_coords.min(axis=0)  # (3,)
            coords_max = real_coords.max(axis=0)  # (3,)
            coords_range = coords_max - coords_min  # (3,)

            # Avoid division by zero
            coords_range[coords_range == 0] = 1

            # Normalize entire sample including padding using real atoms' min/max
            coords_normalized = (coords - coords_min) / coords_range
            
            # Reapply black padding to padded atoms
            if num_real < coords.shape[0]:
                coords_normalized[num_real:] = 0.0

            # Reshape from (28224, 3) to (168, 168, 3) then transpose to (3, 168, 168)
            coords_reshaped = coords_normalized.reshape(self.target_size, self.target_size, 3)
            img = np.transpose(coords_reshaped, (2, 0, 1)).astype(np.float32, copy=False)
            img_batch_list.append(img)
        
        # Return as list if input was list (more memory efficient), else as array
        if is_list:
            return img_batch_list
        else:
            return np.stack(img_batch_list, axis=0)
