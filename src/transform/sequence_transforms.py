"""Preprocessing transforms for padded atom-coordinate sequences."""

from safetensors import torch

from src.transform.ListScrambler import ListScrambler
from src.transform.Padder import Padder
from src.training.LabelEncoder import LabelEncoder

_scrambler = None
_padder = None
_encoder = None

def coordinate_normalization(inputs):
    """Center and standardize each coordinate sequence independently."""
    inputs = inputs - inputs.mean(dim=1, keepdim=True)
    inputs = inputs / (inputs.std(dim=1, keepdim=True) + 1e-6)
    return inputs

def _get_sequence_transforms():
    """Lazily construct and reuse sequence scrambler, padder, and encoder."""
    global _scrambler, _padder, _encoder
    if _scrambler is None:
        _scrambler = ListScrambler(diameter=140)
        _padder = Padder(target_size=168, fill=0)
        _encoder = LabelEncoder()
    return _scrambler, _padder, _encoder


def apply_sequence_transform(examples_data, examples_labels, _real_nr_atoms):
    """Pad, scramble, normalize, and encode a batch of sequence examples."""
    scrambler, padder, encoder = _get_sequence_transforms()
    examples_data = padder(examples_data)
    examples_data = scrambler(examples_data, return_numpy=True, real_nr_atoms=_real_nr_atoms)
    examples_data = coordinate_normalization(torch.tensor(examples_data, dtype=torch.float32))
    examples_labels = encoder.encode_labels(examples_labels)
    return {"data": examples_data, "labels": examples_labels}


def apply_sequence_transform_noscramble(examples_data, examples_labels, _real_nr_atoms):
    """Pad and encode sequence examples without geometric augmentation."""
    _, padder, encoder = _get_sequence_transforms()
    examples_data = padder(examples_data)
    examples_labels = encoder.encode_labels(examples_labels)
    return {"data": examples_data, "labels": examples_labels}
