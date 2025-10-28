from numba import jit
from numpy import ndarray
import numpy as np
import time
from ..utils import randomGenerator

class Scrambler:

    def __init__(self):
        self.random_generator = randomGenerator.RandomGenerator()
    def scramble(self, frame: ndarray, diameter: float) -> ndarray:
        """
        Scramble both the position and orientation of each frame
        """
        frame = self.oriental_scramble(frame)
        frame = self.positional_scramble(frame, diameter)
        return frame
    def oriental_scramble(self, frame: ndarray) -> ndarray:
        """
        Scramble the orientation of each frame
        """
        # Generate random vector outside JIT
        random_vector = self.random_generator.generate_random_unit_vector()

        # Use JIT-compiled implementation
        return self._scramble_orientation(frame, random_vector)
    @staticmethod
    def _scramble_orientation( frame, random_vector) -> ndarray:
        """
        scramble the orientation of each frame (randomly)
        by aligning a vector defined by two arbitrarily chosen atoms (preferably on an axis connecting the intracellular and extracellular ends of the GPCR) to a random unit vector in spheri
        :param frame: np.ndarray, shape=(n_atoms, 3)
            A two dimensional numpy array, with the cartesian coordinates of each atoms.
        :return:
        """

        # Define two arbitrary atoms, here we choose the first two atoms in the frame such that they are always the same atoms
        atom1 = frame[0]
        atom2 = frame[1]
        # Define the vector between the two atoms
        vector = atom2 - atom1
        vector /= np.linalg.norm(vector)  # Normalize the vector
        # Generate a random unit vector
        random_vector /= np.linalg.norm(random_vector)  # Normalize the random vector
        # Compute the rotation matrix using the axis-angle representation
        v = np.cross(vector, random_vector)
        c = np.dot(vector, random_vector) # Cosine of the angle between the vectors
        s = np.linalg.norm(v) # Sine of the angle between the vectors
        if s == 0:
            return frame  # No rotation needed if vectors are parallel
        v /= s  # Normalize the rotation axis

        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s * s))
        # Apply the rotation to all atoms in the frame
        frame = np.dot(frame, rotation_matrix.T)
        return frame

    def positional_scramble(self, frame: ndarray, diameter: float) -> ndarray:
        """
        Scramble the position of each frame
        """
        # Generate random point outside JIT
        radius = diameter / 2
        random_point = self.random_generator.random_uniform(-radius, radius, size=3)

        # Use JIT-compiled implementation
        return self._scramble_position(frame, random_point)
    @staticmethod
    def _scramble_position(frame, random_point: ndarray) -> ndarray:
        """
        scramble the position of each frame (randomly)
        the positional scrambling of frames from a trajectory by moving the centers of mass
        to a randomly sampled coordinate within a sphere of diameter with the size of the largest dimension of the receptor
        :param frame: np.ndarray, shape=(n_atoms, 3)
            A two dimensional numpy array, with the cartesian coordinates of each atoms.
        :return:
        """

        # calculate the center of mass of the frame
        com = frame.mean(axis=0)
        # move the center of mass to the random point
        translation_vector = random_point - com
        frame += translation_vector
        return frame



