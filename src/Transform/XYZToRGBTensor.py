import torch
import numpy as np

class XYZToRGBTensor:
    """Convert 3D molecular coordinates directly to RGB tensor"""

    def __init__(self, target_size=168):
        self.target_size = target_size

    def __call__(self, coords_batch, num_real_atoms_batch=None):
        """
        coords_batch: list of numpy arrays (padded to 28224, 3)
        num_real_atoms_batch: list of original atom counts before padding (for correct normalization)
        returns: tensor of shape (batch, 3, 168, 168)
        """
        # Handle both single sample and batch
        if isinstance(coords_batch, list):
            coords_array = np.array(coords_batch, dtype=np.float32)  # (batch, 28224, 3)
            is_list = True
        else:
            coords_array = np.array(coords_batch, dtype=np.float32)
            is_list = False

        # Handle single sample case
        if coords_array.ndim == 2:
            coords_array = coords_array[np.newaxis, ...]  # Add batch dimension
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = coords_array.shape[0]

        # Normalize only based on real (non-padded) atoms
        coords_normalized = coords_array.copy()

        for i in range(batch_size):
            # Get the number of real atoms for this sample
            num_real = num_real_atoms_batch[i] if num_real_atoms_batch else coords_array.shape[1]

            # Compute min/max only from real atoms (excluding padding)
            real_coords = coords_array[i, :num_real]
            coords_min = real_coords.min(axis=0)  # (3,)
            coords_max = real_coords.max(axis=0)  # (3,)
            coords_range = coords_max - coords_min  # (3,)

            # Avoid division by zero
            coords_range[coords_range == 0] = 1

            # Normalize entire sample including padding using real atoms' min/max
            coords_normalized[i] = (coords_array[i] - coords_min) / coords_range

            # Reapply black padding to  padded atoms
            if num_real < coords_array.shape[1]:
                coords_normalized[i, num_real:] = 0.0


        # Reshape each sample from (28224, 3) to (168, 168, 3)
        coords_reshaped = coords_normalized.reshape(batch_size, self.target_size, self.target_size, 3)
        # Convert to tensor and permute to (batch, 3, 168, 168)
        img_tensor = torch.from_numpy(coords_reshaped).permute(0, 3, 1, 2)  # (batch, 3, 168, 168)

        if squeeze_output:
            img_tensor = img_tensor.squeeze(0)
            return img_tensor

        if is_list:
            return list(img_tensor)
        return img_tensor
