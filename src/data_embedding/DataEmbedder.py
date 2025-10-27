from numpy import ndarray
from src.data_embedding.Scrambler import PositionScrambler, OrientationScrambler
from src.utils.utils import calculate_image_size
from numba import jit, prange
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
class DataEmbedder:
    def __init__(self):
        self.pos_scrambler = PositionScrambler()
        self.ori_scrambler = OrientationScrambler()
        self.time_spent_embedding = 0

    def scramble_frame(self, frame) -> ndarray:
        """
        The scrambling protocol applied to the data prior to submission to the neural network (NN) is
    an unbiasing step in which the position of each frame and its orientation are scrambled (randomly;
    see Figure 3 for more information on the trajectory scrambling). This is undertaken in order to eliminate
    from consideration by the NN any differences among frames that originate not from the time-dependent
    molecular dynamics, but from changes in position or orientation of the ligand-GPCR complex. Thus,
    the scrambling directs the NN algorithm to consider only the intramolecular changes of the protein
    induced by the ligands. This scrambling is introduced in our protocol to achieve the same unbiasing
    that is attained in image classification tasks by random orientation of objects in pictures (which forces
    the object recognition neural networks to understand the shapes and colors of objects, independent of
    their background and orientation).
    -- A Machine Learning Approach for the Discovery of Ligand-Specific Functional Mechanisms of GPCRs
        :param frame: np.ndarray, shape=(n_atoms, 3)
            A two dimensional numpy array, with the cartesian coordinates of each atoms.
        :return:
        """
        frame = self.pos_scrambler.scramble(frame)
        frame = self.ori_scrambler.scramble(frame)
        return frame

    def frame_embedding(self, frame) -> ndarray:
        """
        Embed this frame (xyz coordinates) into an image representation (xyz -> rgb)

        :param frame:
        :return:
        """
        width = height = 180
        frame = self.scramble_frame(frame)
        image = self.create_image(frame, width, height)
        # time spent embedding in seconds
        return image





    @staticmethod
    @jit(nopython=True, parallel=True, fastmath=True)
    def create_image(frame, width, height) -> ndarray:
        """
        Create an image from the frame (xyz coordinates)
        :param frame:
        :param width:
        :param height:
        :return:
        """
        n_atoms = frame.shape[0]
        n_pixels = height * width
        pixels_to_fill = min(n_atoms, n_pixels)
        # Fill the image with zeros (black)
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image_flat = image.reshape(-1, 3)

        if pixels_to_fill == 0:
            return image
        # fill the image with the frame data
        frame_subset = frame[:pixels_to_fill]

        # Precompute min and max for all channels
        min_vals = np.array([frame_subset[:, i].min() for i in range(3)])
        max_vals = np.array([frame_subset[:, i].max() for i in range(3)])
        for i in range(3):
            if max_vals[i] > min_vals[i]:
                normalized = (frame_subset[:, i] - min_vals[i]) / (max_vals[i] - min_vals[i]) * 255
                image_flat[:pixels_to_fill, i] = normalized.astype(np.uint8)
        return image


