from dotenv import load_dotenv
import torch

from src.models.LongSequenceAtomTransformer import LongSequenceAtomTransformer
from src.training.batch_preprocessing import prepare_sequence_batch
from src.transform.sequence_transforms import (
    apply_sequence_transform,
    apply_sequence_transform_noscramble,
)
from src.config.training_config import TrainingConfig
from src.training.training_setup import train_model


def build_weighted_cross_entropy():
    class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.8], dtype=torch.float32)
    return torch.nn.CrossEntropyLoss(weight=class_weights)


if __name__ == "__main__":
    # load dotenv variables
    load_dotenv()

    config_transformerpp = TrainingConfig(
        model_class=LongSequenceAtomTransformer,
        model_args={
        },
        criterion=build_weighted_cross_entropy,
        dataset_location="Sam-Roggeman/SamRoggeman_Thesis_Dataset_full",
        training_transform=apply_sequence_transform,
        validation_transform=apply_sequence_transform_noscramble,
        batch_preparation_fn=prepare_sequence_batch,
        batch_size=64,
        compile_model=True,
        use_mixed_precision=False,
        amp_dtype="float16",
        dataset_size=0.01,
        learning_rate=3e-4,
        max_nr_epochs=10,
        time_limit=3600*3.6,  # 3.6 hours
    )

    train_model(config_transformerpp, "LongSequenceAtomTransformer_Randomsplit_Dataset")
