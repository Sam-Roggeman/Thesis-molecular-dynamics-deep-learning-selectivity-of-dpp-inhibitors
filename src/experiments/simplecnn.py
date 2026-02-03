import os

from huggingface_hub import HfApi
from numba import typeof

from src.Models.OneLayer import OneLayerNet
from src.Models.SimpleCNN import SimpleCNN
from src.utils.training_setup import train_model, load_and_prepare_facehub_datasets
from src.Transform.tranformators import apply_image_transform, apply_image_transform_noscramble
from src.utils.training_config import TrainingConfig
import datasets
if __name__ == "__main__":
    config_scnn = TrainingConfig(
        model_class=SimpleCNN,
        model_args={
            "input_size": 168,
            "dropout_rate": 0.5
        },
        time_limit=4 * 60 * 60,  # 4 hours
        weight_decay=1e-2,
        learning_rate=1e-4,
        batch_size=32,
        patience=15,
        max_nr_epochs=100,
        dataset_location="Sam-Roggeman/SamRoggeman_Thesis_Dataset",
        dataset_size=0.15,
        training_transorm=apply_image_transform,
        validation_transform=apply_image_transform_noscramble,
        transform_batch_size=64,
        transform_num_proc=8,
    )

    train_model(config_scnn, "SimpleCNN_Randomsplit_Dataset")