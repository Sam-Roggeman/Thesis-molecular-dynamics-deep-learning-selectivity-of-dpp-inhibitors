from dataclasses import dataclass, field, asdict
import os
from typing import Callable
import json

import torch
from torch import optim
from torch.nn import CrossEntropyLoss

from src.Transform.tranformators import apply_image_transform, apply_image_transform_noscramble

def calculate_num_cpus():
    """Calculate the number of CPUs to use based on environment variable or default"""
    if "GPULAB_CPUS_RESERVED" in os.environ:
        cpu_count = len(os.environ["GPULAB_CPUS_RESERVED"].split(","))
    else: 
        cpu_count = os.cpu_count()
    return max(1, cpu_count // 2)  # Use half of the reserved CPUs so hyperthreading is enabled, but at least 1
@dataclass
class TrainingConfig:
    """Configuration for training runs"""
    # Model class constructor and keyword arguments used to instantiate it.
    model_class: type = None
    model_args: dict = None

    # Maximum wall-clock training time in seconds.
    time_limit: int = 4 * 60 * 60
    # L2 regularization strength used by AdamW.
    weight_decay: float = 1e-2
    # Base learning rate used by the optimizer.
    learning_rate: float = 1e-4
    # Number of samples per optimization step.
    batch_size: int = 64
    # Early stopping patience (number of eval windows without improvement).
    patience: int = 15
    # Upper bound on full epochs.
    max_nr_epochs: int = 100
    # Optional hard cap on training steps; if None, epoch-based stopping is used.
    max_train_steps: int | None = None
    # Number of train steps to run per epoch abstraction.
    steps_per_epoch: int = 1000
    # Evaluate validation metrics every N train steps.
    eval_every_steps: int = 1000
    # Log training metrics every N train steps.
    log_every_steps: int = 1

    # Loss function factory/callable.
    criterion: Callable = CrossEntropyLoss
    # Optimizer class/factory.
    optimizer: Callable  = optim.AdamW
    # Keyword arguments passed when creating the optimizer.
    optimizer_params: dict = field(default_factory=lambda: {"lr": 1e-4, "weight_decay": 1e-2})
    # Optional scheduler factory/callable.
    scheduler: Callable = None


    # Hugging Face dataset ID or local dataset path.
    dataset_location: str = "Sam-Roggeman/SamRoggeman_Thesis_Dataset"
    # Fraction of each split to use when < 1.0.
    dataset_size: float = 0.01
    # Enable streaming for the train split.
    stream_train_split: bool = True
    # Enable streaming for the validation split.
    stream_validation_split: bool = True
    # Enable streaming for the test split.
    stream_test_split: bool = True
    # Shuffle buffer size used for streamed train data.
    shuffle_buffer_size: int = 100 
    # RNG seed used by dataset shuffle.
    shuffle_seed: int = 42
    # Optional cap on validation batches per evaluation.
    validation_max_batches: int | None = 100
    # Optional cap on test batches.
    test_max_batches: int | None = 100
    # Batch transform applied to training data.
    training_transorm: Callable = apply_image_transform
    # Batch transform applied to validation/test data.
    validation_transform: Callable = apply_image_transform_noscramble
    # Batch size used inside dataset.map for preprocessing.
    transform_batch_size:int = 64
    # CPU workers used by dataset processing/DataLoader. Derived from GPULAB_CPUS_RESERVED when available.
    num_cpus:int = field(default_factory=calculate_num_cpus)

    

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
    # Number of samples per batch during evaluation.
    batch_size: int = 32
    # Fraction of the dataset split to evaluate.
    dataset_size: float = 0.15
    # Loss function callable used for reporting evaluation loss.
    critrion: Callable = CrossEntropyLoss
    # Transform applied to validation/test batches.
    validation_transform: Callable = apply_image_transform_noscramble
    # Batch size used inside dataset.map during preprocessing.
    transform_batch_size:int = 32
    # Number of CPU workers used by the evaluation pipeline.
    num_cpus:int = 8
    # Hugging Face dataset ID or local dataset path.
    dataset_location: str = "Sam-Roggeman/SamRoggeman_Thesis_Dataset"


