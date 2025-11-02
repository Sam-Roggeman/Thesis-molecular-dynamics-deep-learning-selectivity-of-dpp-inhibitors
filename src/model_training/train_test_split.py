from src.utils.configParser import ConfigParser
import os
import shutil
from tqdm import tqdm
import torch
import torchvision

def copy_dataset(dataset, split_path):
    # create the split_path or clear it if it exists
    if os.path.exists(split_path):
        shutil.rmtree(split_path)
    os.makedirs(split_path, exist_ok=True)


    for idx in tqdm(range(len(dataset)), desc=f"Copying {split_path}"):
        # Get the original file path from dataset
        original_idx = dataset.indices[idx]
        original_path, label = dataset.dataset.samples[original_idx]

        # Get class name from path
        class_name = os.path.basename(os.path.dirname(original_path))

        # Create class directory
        class_dir = os.path.join(split_path, class_name)
        os.makedirs(class_dir, exist_ok=True)

        # Copy file
        filename = os.path.basename(original_path)
        dest_path = os.path.join(class_dir, filename)
        shutil.copy2(original_path, dest_path)

def save_image_datasets_to_folders(train_dataset, test_dataset, val_dataset, output_dir):
    """Save image datasets"""
    copy_dataset(train_dataset, os.path.join(output_dir, 'train'))
    copy_dataset(test_dataset, os.path.join(output_dir, 'test'))
    copy_dataset(val_dataset, os.path.join(output_dir, 'val'))

def train_test_val_split():
    """
    Split the data into train, test and validation sets
    :return:
    """
    config_parser  = ConfigParser("config.ini")

    for ds_size in ["small", "medium", "large"]:
        full_dataset_folder = f"./data/dataset/images/{ds_size}/full_dataset"
        dataset_folder = f"./data/dataset/images/{ds_size}/"

        train_size = float(config_parser.get("Train Validation Test Split", "Train Percent"))
        test_size = float(config_parser.get("Train Validation Test Split", "Test Percent"))
        val_size = 1.0 - train_size - test_size
        # split the data into train, test and validation sets
        dataset = torchvision.datasets.ImageFolder(root=full_dataset_folder)

        train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size, val_size])

        # Save the datasets to their respective folders
        save_image_datasets_to_folders(train_dataset, test_dataset, val_dataset, output_dir=dataset_folder)

if __name__ == '__main__':
    train_test_val_split()