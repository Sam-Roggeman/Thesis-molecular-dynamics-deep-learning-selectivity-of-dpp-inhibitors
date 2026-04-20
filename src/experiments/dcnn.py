import os

from huggingface_hub import HfApi

from src.utils.training_setup import train_model
from src.Models.DCNN import CustomDenseNet
from src.Transform.tranformators import apply_image_transform, apply_image_transform_noscramble
from src.utils.training_config import TrainingConfig
import datasets
import os
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()

    config_dcnn = TrainingConfig(
        model_class=CustomDenseNet,
        model_args={
            "growth_rate": 48,
            "block_config": (6, 12, 36, 24),  # 4 dense blocks with 6, 12, 36, 24 layers
            "num_init_features": 96,  # 96 initial filters
            "reduction_ratio": 0.5,  # reduction ratio of 0.5
            "num_classes": 5, 
            "dropout_rate": 0.5,  # dropout rate in the classifier layer
            "feature_dropout_rate": 0.15,  # no dropout in dense layers
            "transition_dropout_rate": 0.1,  # no dropout in transition layers
        },
        dataset_location="Sam-Roggeman/SamRoggeman_Thesis_Dataset_full",
        training_transform=apply_image_transform,
        validation_transform=apply_image_transform_noscramble,
        batch_size=256,
        dataset_size=1.0,  # Use the full dataset
    )
    train_model(config_dcnn, "CustomDenseNet_Randomsplit_Dataset")
