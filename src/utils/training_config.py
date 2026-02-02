from dataclasses import dataclass, field
import os
from typing import Callable
import json
from torch import optim
from torch.nn import CrossEntropyLoss

from src.Transform.tranformators import apply_image_transform, apply_image_transform_noscramble

@dataclass
class TrainingConfig:
    """Configuration for training runs"""
    # Model class and its arguments
    model_class: type
    model_args: dict

    # Training loop parameters
    time_limit: int = 4 * 60 * 60
    weight_decay: float = 1e-2
    learning_rate: float = 1e-4
    batch_size: int = 16
    patience: int = 15
    max_nr_epochs: int = 100

    criterion: Callable = CrossEntropyLoss
    optimizer: Callable  = optim.AdamW
    optimizer_params: dict = field(default_factory=lambda: {"lr": 1e-4, "weight_decay": 1e-2})
    scheduler: Callable = None


    # Dataset parameters
    dataset_location: str = "Sam-Roggeman/SamRoggeman_Thesis_Dataset"
    dataset_size: float = 0.15
    training_transorm: Callable = apply_image_transform
    validation_transform: Callable = apply_image_transform_noscramble
    transform_batch_size:int = 32
    transform_num_proc:int = 8

    def save(self, path):
        """
        Save the configuration to the specified path in json format.
        :param path:  Path to save the configuration.
        """
        os.makedirs(path, exist_ok=True)
        config_path = os.path.join(path, "training_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
    def __init__(self, filepath=None, **kwargs):
        if filepath:
            self.load(filepath)
        for key, value in kwargs.items():
            setattr(self, key, value)
    def load(self, path):
        """
        Load the configuration from a json file.
        :param path: Path to the configuration file.
        """
        config_path = os.path.join(path, "training_config.json")
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
            self.__dict__.update(config_dict)

@dataclass
class TestConfig():
    """Configuration for validation runs"""
    batch_size: int = 32
    dataset_size: float = 0.15
    critrion: Callable = CrossEntropyLoss
    validation_transform: Callable = apply_image_transform_noscramble
    transform_batch_size:int = 32
    transform_num_proc:int = 8

