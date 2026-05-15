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

    def __call__(self, batch, return_numpy=False, real_nr_atoms=None):
        """
        Apply scrambling to a batch efficiently
        Args:
            batch: list of arrays or numpy array of shape (batch, n_atoms, 3)
            return_numpy: if True, return list of numpy arrays (faster); if False, return list of lists
            real_nr_atoms: optional per-sample atom counts. If provided, only the first
                ``real_nr_atoms[i]`` rows of each sample are scrambled and any padded
                rows after that are left untouched.
        Returns:
            List of scrambled frames (as numpy arrays if return_numpy=True, else as lists)
        """
        # Ensure input is list of arrays
        if isinstance(batch, np.ndarray):
            batch_list = [batch[i] for i in range(batch.shape[0])]
        else:
            batch_list = [np.array(frame, dtype=np.float32) if not isinstance(frame, np.ndarray) else frame for frame in batch]

        if real_nr_atoms is not None and len(real_nr_atoms) != len(batch_list):
            raise ValueError(
                f"real_nr_atoms length ({len(real_nr_atoms)}) must match batch size ({len(batch_list)})."
            )
        
        # Process each frame
        scrambled_batch = []
        for idx, frame_array in enumerate(batch_list):
            n_real_atoms = None if real_nr_atoms is None else int(real_nr_atoms[idx])
            frame_array = self._scramble_single_frame(frame_array, n_real_atoms=n_real_atoms)
            # Keep as numpy if return_numpy=True, else convert to list
            if not return_numpy:
                frame_array = frame_array.tolist()
            scrambled_batch.append(frame_array)

        return scrambled_batch

    def _scramble_single_frame(self, frame, n_real_atoms=None):
        """Scramble only the real atom prefix of one frame and preserve padding rows."""
        frame = np.asarray(frame, dtype=np.float32)

        if n_real_atoms is None:
            n_real_atoms = frame.shape[0]

        n_real_atoms = max(0, min(int(n_real_atoms), frame.shape[0]))
        if n_real_atoms == 0:
            return frame.copy()

        real_atoms = frame[:n_real_atoms].copy()
        padded_atoms = frame[n_real_atoms:].copy() if n_real_atoms < frame.shape[0] else None

        if n_real_atoms >= 2:
            real_atoms = self._oriental_scramble_single(real_atoms)

        real_atoms = self._positional_scramble_vectorized(real_atoms)

        if padded_atoms is None:
            return real_atoms
        return np.concatenate([real_atoms, padded_atoms], axis=0)

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