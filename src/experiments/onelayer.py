import os

from huggingface_hub import HfApi

from src.Models.OneLayer import OneLayerNet
from src.utils.training_setup import train_model, load_and_prepare_facehub_datasets
from src.Transform.tranformators import apply_image_transform, apply_image_transform_noscramble
from src.utils.training_config import TrainingConfig
import datasets
if __name__ == "__main__":
    config_dcnn = TrainingConfig(
        model_class=OneLayerNet,
        model_args={
            "input_size": 3 * 168 * 168,
            "nr_neurons": 512,
            "output_size": 5,
            "dropout_rate": 0.5
        },
        time_limit=4 * 60 * 60,  # 4 hours
        weight_decay=1e-2,
        learning_rate=1e-4,
        batch_size=32,
        patience=15,
        max_nr_epochs=100,
        dataset_location="Sam-Roggeman/SamRoggeman_Thesis_Dataset_full",
        dataset_size=0.15,
        training_transorm=apply_image_transform,
        validation_transform=apply_image_transform_noscramble,
        transform_batch_size=64,
    )

    train_model(config_dcnn, "OneLayerNet_Randomsplit_Dataset")