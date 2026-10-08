"""Torch geometric augmentation for molecular coordinate frames."""

import torch


class TorchScrambler:
    """Randomly rotate and translate single frames or batches of frames.

    Inputs use ``[N, 3]`` or ``[B, N, 3]`` layout. Rotation aligns the vector
    between the first two atoms with a random direction; translation moves the
    frame center to a random point inside the configured diameter.
    """

    def __init__(self, diameter: float, device='cpu', seed =42):
        """Configure the translation volume, device, and global random seed."""
        self.diameter = diameter
        self.device = device
        torch.manual_seed(seed)


    def __call__(self, frame: torch.Tensor, batched) -> torch.Tensor:
        """
        Apply rotation and translation while preserving the input rank.
        Args:
            frame: list of shape (n_atoms, 3) or (batch, n_atoms, 3)
        Args:
            batched: Retained for transform-pipeline compatibility; rank is inferred
                from ``frame``.
        Returns:
            Scrambled frame with the same shape as the input.
        """
        # depth of the nested list
        if frame.dim() == 2:
            # Single frame
            frame = self.oriental_scramble(frame)
            frame = self.positional_scramble(frame)
        elif frame.dim() == 3:
            # Batch of frames
            frame = self.oriental_scramble_batch(frame)
            frame = self.positional_scramble_batch(frame)
        return frame

    def oriental_scramble(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Scramble orientation of a single frame
        Args:
            frame: torch.Tensor of shape (n_atoms, 3)
        """
        # Generate random unit vector
        random_vector = torch.randn(3, device=self.device)
        random_vector = random_vector / torch.norm(random_vector)

        # Get vector between first two atoms
        atom1 = frame[0]
        atom2 = frame[1]
        vector = atom2 - atom1
        vector = vector / torch.norm(vector)

        # Compute rotation matrix
        rotation_matrix = self._compute_rotation_matrix(vector, random_vector)

        # Apply rotation
        frame = torch.matmul(frame, rotation_matrix.T)
        return frame

    def oriental_scramble_batch(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Scramble orientation of a batch of frames
        Args:
            frames: torch.Tensor of shape (batch, n_atoms, 3)
        """
        batch_size = frames.shape[0]

        # Generate random unit vectors for each frame in batch
        random_vectors = torch.randn(batch_size, 3, device=self.device)
        random_vectors = random_vectors / torch.norm(random_vectors, dim=1, keepdim=True)

        # Get vectors between first two atoms for each frame
        atom1 = frames[:, 0, :]  # (batch, 3)
        atom2 = frames[:, 1, :]  # (batch, 3)
        vectors = atom2 - atom1
        vectors = vectors / torch.norm(vectors, dim=1, keepdim=True)

        # Compute rotation matrices for batch
        rotation_matrices = self._compute_rotation_matrix_batch(vectors, random_vectors)

        # Apply rotations: (batch, n_atoms, 3) @ (batch, 3, 3)
        frames = torch.bmm(frames, rotation_matrices.transpose(1, 2))
        return frames

    def _compute_rotation_matrix(self, vector: torch.Tensor, random_vector: torch.Tensor) -> torch.Tensor:
        """
        Compute rotation matrix using axis-angle representation
        Args:
            vector: torch.Tensor of shape (3,)
            random_vector: torch.Tensor of shape (3,)
        Returns:
            rotation_matrix: torch.Tensor of shape (3, 3)
        """
        # Cross product for rotation axis
        v = torch.cross(vector, random_vector)
        c = torch.dot(vector, random_vector)  # Cosine of angle
        s = torch.norm(v)  # Sine of angle

        if s < 1e-10:
            # No rotation needed if vectors are parallel
            return torch.eye(3, device=self.device)

        v = v / s  # Normalize rotation axis

        # Skew-symmetric matrix
        kmat = torch.tensor([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ], device=self.device)

        # Rodrigues' rotation formula
        rotation_matrix = (torch.eye(3, device=self.device) +
                           kmat +
                           torch.matmul(kmat, kmat) * ((1 - c) / (s * s)))
        return rotation_matrix

    def _compute_rotation_matrix_batch(self, vectors: torch.Tensor, random_vectors: torch.Tensor) -> torch.Tensor:
        """
        Compute rotation matrices for a batch
        Args:
            vectors: torch.Tensor of shape (batch, 3)
            random_vectors: torch.Tensor of shape (batch, 3)
        Returns:
            rotation_matrices: torch.Tensor of shape (batch, 3, 3)
        """
        batch_size = vectors.shape[0]

        # Cross product for rotation axes
        v = torch.cross(vectors, random_vectors, dim=1)  # (batch, 3)
        c = (vectors * random_vectors).sum(dim=1)  # (batch,) - cosine of angles
        s = torch.norm(v, dim=1)  # (batch,) - sine of angles

        # Handle parallel vectors
        mask = s > 1e-10
        v[mask] = v[mask] / s[mask].unsqueeze(1)

        # Create skew-symmetric matrices for batch
        kmat = torch.zeros(batch_size, 3, 3, device=self.device)
        kmat[:, 0, 1] = -v[:, 2]
        kmat[:, 0, 2] = v[:, 1]
        kmat[:, 1, 0] = v[:, 2]
        kmat[:, 1, 2] = -v[:, 0]
        kmat[:, 2, 0] = -v[:, 1]
        kmat[:, 2, 1] = v[:, 0]

        # Rodrigues' rotation formula for batch
        eye = torch.eye(3, device=self.device).unsqueeze(0).expand(batch_size, -1, -1)
        kmat_squared = torch.bmm(kmat, kmat)

        # Compute rotation factor
        rotation_factor = ((1 - c) / (s * s + 1e-10)).unsqueeze(1).unsqueeze(2)

        rotation_matrices = eye + kmat + kmat_squared * rotation_factor

        # For parallel vectors, use identity
        rotation_matrices[~mask] = torch.eye(3, device=self.device)

        return rotation_matrices

    def positional_scramble(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Scramble position of a single frame
        Args:
            frame: torch.Tensor of shape (n_atoms, 3)
        """
        radius = self.diameter / 2
        random_point = (torch.rand(3, device=self.device) * 2 - 1) * radius

        # Center of mass
        com = frame.mean(dim=0)

        # Translation vector
        translation_vector = random_point - com
        frame = frame + translation_vector
        return frame

    def positional_scramble_batch(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Scramble position of a batch of frames
        Args:
            frames: torch.Tensor of shape (batch, n_atoms, 3)
        """
        batch_size = frames.shape[0]
        radius = self.diameter / 2

        # Generate random points for each frame
        random_points = (torch.rand(batch_size, 3, device=self.device) * 2 - 1) * radius

        # Center of mass for each frame
        com = frames.mean(dim=1)  # (batch, 3)

        # Translation vectors
        translation_vectors = random_points - com  # (batch, 3)

        # Apply translation
        frames = frames + translation_vectors.unsqueeze(1)  # Broadcast over n_atoms
        return frames


# Create a custom transform
class ScramblingTransform:
    """Small transform wrapper exposing a configured ``TorchScrambler``."""

    def __init__(self, diameter: float, device='cpu'):
        self.scrambler = TorchScrambler(diameter, device=device)

    def __call__(self, frame):
        return self.scrambler(frame)

