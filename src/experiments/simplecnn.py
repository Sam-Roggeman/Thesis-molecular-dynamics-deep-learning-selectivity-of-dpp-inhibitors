import os

from huggingface_hub import HfApi

from src.Models.SimpleCNN import SimpleCNN
from src.utils.training_setup import train_model
from src.Transform.tranformators import apply_image_transform, apply_image_transform_noscramble
from src.utils.training_config import TrainingConfig
if __name__ == "__main__":
    config_scnn = TrainingConfig(
        model_class=SimpleCNN,
        model_args={
            "input_size": 168,
            "dropout_rate": 0.2
        },
        dataset_location="Sam-Roggeman/SamRoggeman_Thesis_Dataset_full",
        training_transform=apply_image_transform,
        validation_transform=apply_image_transform_noscramble,
        batch_size=128*4,
        time_limit=24 * 60 * 60,  # 24 hours
    )

    train_model(config_scnn, "SimpleCNN_Randomsplit_Dataset")