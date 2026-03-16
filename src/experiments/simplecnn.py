import os

from huggingface_hub import HfApi
from numba import typeof

from src.Models.SimpleCNN import SimpleCNN
from src.utils.training_setup import train_model, load_and_prepare_facehub_datasets
from src.Transform.tranformators import apply_image_transform, apply_image_transform_noscramble
from src.utils.training_config import TrainingConfig
if __name__ == "__main__":
    config_scnn = TrainingConfig(
        model_class=SimpleCNN,
        model_args={
            "input_size": 168,
            "dropout_rate": 0.5
        },
        time_limit=16 * 60 * 60,  # 16 hours
        weight_decay=1e-2,
        learning_rate=1e-4,
        batch_size=256,
        patience=50,
        max_nr_epochs=500,
        dataset_location="Sam-Roggeman/SamRoggeman_Thesis_Dataset_full",
        dataset_size=0.001,
        training_transorm=apply_image_transform,
        validation_transform=apply_image_transform_noscramble,
        transform_batch_size=64,
    )

    train_model(config_scnn, "SimpleCNN_Randomsplit_Dataset")