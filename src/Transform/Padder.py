import torchvision.transforms as transforms
from torchvision.transforms.functional import pad
import torch
class Padder:
    def __init__(self, target_size=168, fill=0.0):
        self.target_size = target_size
        self.fill = fill

    def __call__(self, data_examples):
        """Pad entries to target_size x target_size by appending pixels to the back"""
        target_pixels = self.target_size * self.target_size
        # Calculate padding to reach target size
        total_pad_pixels = [target_pixels - len(entry) for entry in data_examples]
        # append total_pad_pixels [0,0,0] to the back of the list
        data_examples = [data_examples[i]+[[self.fill]*3] * total_pad for i, total_pad in enumerate(total_pad_pixels)]

        return data_examples

    def reapply_padding(self, examples_data, real_nr_atoms):
        """Reapply padding to a batch of examples based on their real number of atoms"""
        padded_data = []
        for i, data in enumerate(examples_data):
            n_real_atoms = real_nr_atoms[i]
            n_total_pixels = self.target_size * self.target_size
            n_pad_pixels = n_total_pixels - n_real_atoms
            padded_example = data[:n_real_atoms] + [[self.fill]*3] * n_pad_pixels
            padded_data.append(padded_example)
        return padded_data
