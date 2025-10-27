from numpy import ndarray
import numpy as np
import time
from ..utils import randomGenerator

class Scrambler:
    def __init__(self):
        self.time_spent_scrambling = 0

class OrientationScrambler(Scrambler):
    def __init__(self):
        super().__init__()

    def scramble(self, frame) -> ndarray:
        """
        scramble the orientation of each frame (randomly)
        by aligning a vector defined by two arbitrarily chosen atoms (preferably on an axis connecting the intracellular and extracellular ends of the GPCR) to a random unit vector in spheri
        :param frame: np.ndarray, shape=(n_atoms, 3)
            A two dimensional numpy array, with the cartesian coordinates of each atoms.
        :return:
        """
        start = time.time()

        # Define two arbitrary atoms, here we choose the first two atoms in the frame
        atom1 = frame[0]
        atom2 = frame[1]
        # Define the vector between the two atoms
        vector = atom2 - atom1
        vector /= np.linalg.norm(vector)  # Normalize the vector
        # Generate a random unit vector
        random_vector = np.random.normal(size=3)
        random_vector /= np.linalg.norm(random_vector)  # Normalize the random vector
        # Compute the rotation matrix using the axis-angle representation
        v = np.cross(vector, random_vector)
        c = np.dot(vector, random_vector)
        s = np.linalg.norm(v)
        if s == 0:
            return frame  # No rotation needed if vectors are parallel
        v /= s  # Normalize the rotation axis
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        # Apply the rotation to all atoms in the frame
        frame = np.dot(frame, rotation_matrix.T)

        end = time.time()
        self.time_spent_scrambling += end - start
        return frame
class PositionScrambler(Scrambler):
    def __init__(self):
        super().__init__()
    def scramble(self, frame) -> ndarray:
        """
        scramble the position of each frame (randomly)
        the positional scrambling of frames from a trajectory by moving the centers of mass
        to a randomly sampled coordinate within a sphere of diameter 90 Å (the size of the largest dimension of the receptor
        :param frame: np.ndarray, shape=(n_atoms, 3)
            A two dimensional numpy array, with the cartesian coordinates of each atoms.
        :return:
        """
        start = time.time()

        # calculate the center of mass of the frame
        com = frame.mean(axis=0)
        # generate a random point within a sphere of diameter 90 Å
        radius = 45.0  # radius is half of diameter
        random_point = np.random.uniform(-radius, radius, size=3)
        # move the center of mass to the random point
        translation_vector = random_point - com
        frame += translation_vector

        end = time.time()
        self.time_spent_scrambling += end - start

        return frame