from src.Transform.ListScrambler import ListScrambler, ScramblingTransform
from src.Transform.Padder import Padder
from src.Transform.XYZToRGBTensor import XYZToRGBTensor
from src.model_training.LabelEncoder import LabelEncoder
def apply_image_transform(examples_data, examples_labels, real_nr_atoms):
    """Apply transform to each entry in the batch"""
    # Change from XYZ (tensor shape: [28224,3]) to RGB [3,168,168]
    rgb_transformer = XYZToRGBTensor(target_size=168)
    # Scramble with diameter 140A
    scrambler = ScramblingTransform(140)
    # Pad to 168x168 = 28224
    padder = Padder(target_size=168, fill=0)
    encoder = LabelEncoder()

    examples_data = rgb_transformer(padder(scrambler(examples_data)), real_nr_atoms)
    examples_labels = encoder.encode_labels(examples_labels)
    return {"data": examples_data, "labels": examples_labels}

def apply_image_transform_noscramble(examples_data, examples_labels, real_nr_atoms):
    """Apply transform to each entry in the batch"""
    # Change from XYZ (tensor shape: [28224,3]) to RGB [3,168,168]
    rgb_transformer = XYZToRGBTensor(target_size=168)
    # Pad to 168x168 = 28224
    padder = Padder(target_size=168, fill=0)
    encoder = LabelEncoder()

    examples_data = rgb_transformer(padder(examples_data), real_nr_atoms)
    examples_labels = encoder.encode_labels(examples_labels)

    return {"data": examples_data, "labels": examples_labels}