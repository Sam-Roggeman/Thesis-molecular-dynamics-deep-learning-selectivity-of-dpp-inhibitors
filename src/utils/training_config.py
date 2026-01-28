from dataclasses import dataclass, field
import os
from typing import Callable

from huggingface_hub import HfApi
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
    dataset_location: str = "../../data/dataset/full_dataset/"
    hf_token: str = ""
    cache_folder: str = None
    dataset_size: float = 0.15
    training_transorm: Callable = apply_image_transform
    validation_transform: Callable = apply_image_transform_noscramble
    transform_batch_size:int = 32
    transform_num_proc:int = 8

    def __post_init__(self):
        if self.cache_folder is None:
            self.cache_folder = '/home/samro/.cache/huggingface/datasets/sam_roggeman_thesis_dataset/default/0.0.0/bd5fc36a30b06598/'

