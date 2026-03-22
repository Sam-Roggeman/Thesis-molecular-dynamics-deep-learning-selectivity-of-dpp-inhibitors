import numpy as np
from src.Transform.ListScrambler import ListScrambler
from src.Transform.Padder import Padder
from src.Transform.XYZToRGBTensor import XYZToRGBTensor
from src.model_training.LabelEncoder import LabelEncoder

# Singleton instances to avoid repeated instantiation
_rgb_transformer = None
_scrambler = None
_padder = None
_encoder = None

def _get_transforms():
    """Lazy-load singleton transform instances"""
    global _rgb_transformer, _scrambler, _padder, _encoder
    if _rgb_transformer is None:
        _rgb_transformer = XYZToRGBTensor(target_size=168)
        _scrambler = ListScrambler(diameter=140)
        _padder = Padder(target_size=168, fill=0)
        _encoder = LabelEncoder()
    return _rgb_transformer, _scrambler, _padder, _encoder

def apply_image_transform(examples_data, examples_labels, real_nr_atoms):
    """Apply transform to batch: pad → scramble → normalize → RGB tensor. Works with numpy arrays."""
    rgb_transformer, scrambler, padder, encoder = _get_transforms()
    
    # Pad first (in-place numpy)
    examples_data = padder(examples_data)
    # Scramble (stays as numpy arrays, no .tolist() conversion)
    examples_data = scrambler(examples_data, return_numpy=True)
    # Normalize and reshape to RGB tensor
    examples_data = rgb_transformer(examples_data, real_nr_atoms)
    # Encode labels
    examples_labels = encoder.encode_labels(examples_labels)
    
    return {"data": examples_data, "labels": examples_labels}

def apply_image_transform_noscramble(examples_data, examples_labels, real_nr_atoms):
    """Apply transform without scrambling: pad → normalize → RGB tensor."""
    rgb_transformer, _, padder, encoder = _get_transforms()
    
    # Pad first
    examples_data = padder(examples_data)
    # Skip scrambling, go directly to RGB normalization
    examples_data = rgb_transformer(examples_data, real_nr_atoms)
    # Encode labels
    examples_labels = encoder.encode_labels(examples_labels)
    
    return {"data": examples_data, "labels": examples_labels}