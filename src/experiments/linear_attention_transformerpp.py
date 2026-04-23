from dotenv import load_dotenv

from src.Models.LinearAttentionTransformerPP import LinearAttentionTransformerPP
from src.model_training.batch_preprocessing import prepare_sequence_batch
from src.Transform.sequence_transforms import (
    apply_sequence_transform,
    apply_sequence_transform_noscramble,
)
from src.utils.training_config import TrainingConfig
from src.utils.training_setup import train_model


if __name__ == "__main__":
    # load dotenv variables
    load_dotenv()

    config_transformerpp = TrainingConfig(
        model_class=LinearAttentionTransformerPP,
        model_args={
            "input_dim": 3,
            "num_classes": 5,
            "seq_len": 168 * 168,
            "d_model": 64,
            "n_heads": 4,
            "n_layers": 3,
            "ffn_multiplier": 4,
            "dropout": 0.1,
        },
        dataset_location="Sam-Roggeman/SamRoggeman_Thesis_Dataset_full",
        training_transform=apply_sequence_transform,
        validation_transform=apply_sequence_transform_noscramble,
        batch_preparation_fn=prepare_sequence_batch,
        batch_size=64,
        compile_model=True,
        use_mixed_precision=False,
        amp_dtype="float16",
        dataset_size=0.01,
    )

    train_model(config_transformerpp, "LinearAttentionTransformerPP_Randomsplit_Dataset")
