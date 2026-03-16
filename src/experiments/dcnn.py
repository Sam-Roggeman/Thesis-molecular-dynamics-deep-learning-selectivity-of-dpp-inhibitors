import os

from huggingface_hub import HfApi
from numba import typeof

from src.utils.training_setup import train_model, load_and_prepare_facehub_datasets
from src.Models.DCNN import CustomDenseNet
from src.Transform.tranformators import apply_image_transform, apply_image_transform_noscramble
from src.utils.training_config import TrainingConfig
import datasets
import os
from dotenv import load_dotenv

if __name__ == "__main__":

    config_dcnn = TrainingConfig(
        model_class=CustomDenseNet,
        model_args={
            "growth_rate": 48,
            "block_config": (6, 12, 36, 24),  # 4 dense blocks with 6, 12, 36, 24 layers
            "num_init_features": 96,  # 96 initial filters
            "reduction_ratio": 0.5,  # reduction ratio of 0.5
            "num_classes": 5
        },
        time_limit= 4 * 60 * 60,  # 4 hours
        weight_decay=1e-2,
        learning_rate=1e-4,
        batch_size=16,
        patience=15,
        max_nr_epochs=100,
        dataset_location="Sam-Roggeman/SamRoggeman_Thesis_Dataset_full",
        dataset_size=0.15,
        training_transorm=apply_image_transform,
        validation_transform=apply_image_transform_noscramble,
        transform_batch_size=32,
        num_cpus=8,
    )

    train_model(config_dcnn, "CustomDenseNet_Randomsplit_Dataset")