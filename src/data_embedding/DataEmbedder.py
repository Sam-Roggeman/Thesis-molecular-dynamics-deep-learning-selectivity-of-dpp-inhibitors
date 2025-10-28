from numpy import ndarray
from src.utils.utils import calculate_image_size
from numba import jit, prange
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
class DataEmbedder:
    def __init__(self, width, height):
        self.width = width
        self.height = height



    def frame_embedding(self, frame) -> ndarray:
        """
        Embed this frame (xyz coordinates) into an image representation (xyz -> rgb)

        :param frame:
        :return:
        """
        image = self.create_image(frame, self.width, self.height)
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


