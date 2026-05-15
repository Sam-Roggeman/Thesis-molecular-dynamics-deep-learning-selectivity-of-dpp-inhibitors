from src.Transform.ListScrambler import ListScrambler
from src.Transform.Padder import Padder
from src.model_training.LabelEncoder import LabelEncoder

_scrambler = None
_padder = None
_encoder = None


def _get_sequence_transforms():
    global _scrambler, _padder, _encoder
    if _scrambler is None:
        _scrambler = ListScrambler(diameter=140)
        _padder = Padder(target_size=168, fill=0)
        _encoder = LabelEncoder()
    return _scrambler, _padder, _encoder


def apply_sequence_transform(examples_data, examples_labels, _real_nr_atoms):
    scrambler, padder, encoder = _get_sequence_transforms()
    examples_data = padder(examples_data)
    examples_data = scrambler(examples_data, return_numpy=True, real_nr_atoms=_real_nr_atoms)
    examples_labels = encoder.encode_labels(examples_labels)
    return {"data": examples_data, "labels": examples_labels}


def apply_sequence_transform_noscramble(examples_data, examples_labels, _real_nr_atoms):
    _, padder, encoder = _get_sequence_transforms()
    examples_data = padder(examples_data)
    examples_labels = encoder.encode_labels(examples_labels)
    return {"data": examples_data, "labels": examples_labels}
