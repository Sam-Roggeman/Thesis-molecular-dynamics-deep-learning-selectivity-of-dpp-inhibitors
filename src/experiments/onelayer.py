from src.Models.OneLayer import OneLayerNet
from src.Transform.tranformators import apply_image_transform, apply_image_transform_noscramble
from src.utils.training_config import TrainingConfig
from src.utils.training_setup import train_model

def apply_transform_flatten(transform,examples_data, examples_labels, real_nr_atoms):
    transformed = transform(examples_data, examples_labels, real_nr_atoms)
    # transformed is a batch (list) of tensors with shape (batch_size, channels, height, width)
    # Flatten to (batch, channels, height  * width)
    channels, height, width = transformed['data'][0].shape
    transformed['data'] = [data.reshape(channels * height * width) for data in transformed['data']]
    return transformed

def apply_image_transform_flatten(examples_data, examples_labels, real_nr_atoms):
    return apply_transform_flatten(apply_image_transform,examples_data, examples_labels, real_nr_atoms)
def apply_image_transform_noscramble_flatten(examples_data, examples_labels, real_nr_atoms):
    return apply_transform_flatten(apply_image_transform_noscramble,examples_data, examples_labels, real_nr_atoms)

if __name__ == '__main__':
    config_onelayer = TrainingConfig(
        model_class=OneLayerNet,
        model_args={
            "input_size" : 168 * 168 * 3,
            "output_size" : 5,
            "nr_neurons" : 256
        },
        time_limit = 4*60 * 60,  # 4 hours
        weight_decay = 1e-2,
        learning_rate = 1e-4,
        batch_size = 16,
        patience = 15,
        max_nr_epochs = 200,
        dataset_location = "/mnt/x/School/trajects/dataset/full_dataset",
        dataset_size = 0.15,
        training_transorm = apply_image_transform_flatten,
        validation_transform = apply_image_transform_noscramble_flatten,
        transform_batch_size = 32,
        transform_num_proc = 8
    )
    config_onelayer.model_args["nr_neurons"] *= 25
    train_model(config_onelayer, "OneLayerNet100x_Randomsplit_Dataset")
    config_onelayer.learning_rate *= 10
    train_model(config_onelayer, "OneLayerNet100x_LR1e-3_Randomsplit_Dataset")






