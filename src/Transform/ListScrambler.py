import math
import random
import numpy as np
from functools import lru_cache


class ListScrambler:
    """Optimized list-based scrambler for HuggingFace datasets using NumPy"""

    def __init__(self, diameter: float, seed=42):
        self.diameter = diameter
        self.radius = diameter / 2
        random.seed(seed)
        np.random.seed(seed)

    def __call__(self, batch, return_numpy=False):
        """
        Apply scrambling to a batch efficiently
        Args:
            batch: list of arrays or numpy array of shape (batch, n_atoms, 3)
            return_numpy: if True, return list of numpy arrays (faster); if False, return list of lists
        Returns:
            List of scrambled frames (as numpy arrays if return_numpy=True, else as lists)
        """
        # Ensure input is list of arrays
        if isinstance(batch, np.ndarray):
            batch_list = [batch[i] for i in range(batch.shape[0])]
        else:
            batch_list = [np.array(frame, dtype=np.float32) if not isinstance(frame, np.ndarray) else frame for frame in batch]
        
        # Process each frame
        scrambled_batch = []
        for frame_array in batch_list:
            frame_array = self._oriental_scramble_single(frame_array)
            frame_array = self._positional_scramble_vectorized(frame_array)
            # Keep as numpy if return_numpy=True, else convert to list
            if not return_numpy:
                frame_array = frame_array.tolist()
            scrambled_batch.append(frame_array)

        return scrambled_batch

    def _oriental_scramble_single(self, frame):
        """
        Orientation scrambling for a single frame
        Args:
            frame: np.ndarray of shape (n_atoms, 3)
        Returns:
            Scrambled frame
        """
        # Generate random unit vector
        random_vector = np.random.randn(3).astype(np.float32)
        random_vector /= np.linalg.norm(random_vector)

        # Get vector between first two atoms
        vector = frame[1] - frame[0]
        vector /= np.linalg.norm(vector)

        # Compute rotation matrix
        rotation_matrix = self._compute_rotation_matrix_single(vector, random_vector)

        # Apply rotation
        frame = frame @ rotation_matrix.T

        return frame

    def _compute_rotation_matrix_single(self, vector, random_vector):
        """
        Compute rotation matrix using Rodrigues' formula (single frame)
        Args:
            vector: np.ndarray of shape (3,)
            random_vector: np.ndarray of shape (3,)
        Returns:
            rotation_matrix: np.ndarray of shape (3, 3)
        """
        # Cross product
        v = np.cross(vector, random_vector)
        c = np.dot(vector, random_vector)  # cosine
        s = np.linalg.norm(v)  # sine

        if s < 1e-10:
            return np.eye(3, dtype=np.float32)

        v = v / s  # normalize

        # Skew-symmetric matrix
        kmat = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ], dtype=np.float32)

        # Rodrigues' formula
        kmat_sq = kmat @ kmat
        rotation_matrix = np.eye(3, dtype=np.float32) + kmat + kmat_sq * ((1 - c) / (s * s))

        return rotation_matrix

    def _positional_scramble_vectorized(self, frames):
        """
        Positional scrambling for a single frame
        Args:
            frame: np.ndarray of shape (n_atoms, 3)
        Returns:
            Translated frame
        """
        # Generate random point
        random_point = (np.random.rand(3).astype(np.float32) * 2 - 1) * self.radius

        # Center of mass
        com = frames.mean(axis=0)

        # Translation vector
        translation_vector = random_point - com
        frames = frames + translation_vector

        return frames


class ScramblingTransform:
    """HuggingFace-compatible transform with caching"""

    def __init__(self, diameter: float):
        self.scrambler = ListScrambler(diameter)

    def __call__(self, batch):
        """Apply scrambling to a batch from HuggingFace datasets"""
        return self.scrambler(batch)