from dataclasses import dataclass, field, asdict
import os
from typing import Callable
import json

import torch
from torch import optim
from torch.nn import CrossEntropyLoss

from src.Transform.tranformators import apply_image_transform, apply_image_transform_noscramble

@dataclass
class TrainingConfig:
    """Configuration for training runs"""
    # Model class and its arguments
    model_class: type = None
    model_args: dict = None

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
        torch.save(asdict(self), os.path.join(path, "training_config.pt"))
        with open(os.path.join(path, "training_config.json"), "w") as f:
            # Make sure to convert any non-serializable fields to string if necessary
            d = asdict(self)
            d = {k: (str(v) if not isinstance(v, (int, float, str, dict, list, type(None))) else v) for k, v in d.items()}
            json.dump(d, f, indent=4)

    @classmethod
    def load(cls, path):
        data = torch.load(path, weights_only=False)
        return cls(**data)


@dataclass
class TestConfig():
    """Configuration for validation runs"""
    batch_size: int = 32
    dataset_size: float = 0.15
    critrion: Callable = CrossEntropyLoss
    validation_transform: Callable = apply_image_transform_noscramble
    transform_batch_size:int = 32
    transform_num_proc:int = 8
    dataset_location: str = "Sam-Roggeman/SamRoggeman_Thesis_Dataset"


