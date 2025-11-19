from src.utils.configParser import ConfigParser
import os
import shutil
from tqdm import tqdm
import torch
import torchvision

from torch.utils.data import random_split

from src.utils.utils import get_binding_classes


def split_tensor(tensor, split_ratio, shuffle):
    """
    Split a tensor into two parts based on the split ratio
    :param tensor: Input tensor
    :param split_ratio: Ratio to split the tensor
    :return: part1, part2 - two tensors
    """
    if shuffle:
        # shuffle the tensor along the first dimension
        tensor = tensor[torch.randperm(tensor.size(0))]
    num_elements = tensor.size(0)
    split_index = int(num_elements * split_ratio)
    part1 = tensor[:split_index]
    part2 = tensor[split_index:]
    return part1, part2


def create_ligand_splits_1_train_1_valtest(classified_tensors, random_seed=42):
    """
    Create splits such that data from 1 ligand is used for training and data from the other ligand is used for validation and testing
     (50% val, 50% test)
    :param classified_tensors: {class_name: [tensors]}
    :param random_seed: Random seed for reproducibility
    :return: train_dataset, val_dataset, test_dataset - classified_tensors {class_name: [tensors]}
    """
    print('Creating 1 ligand train, 1 ligand val/test splits...')
    torch.manual_seed(random_seed)
    train_dataset = {}
    val_dataset = {}
    test_dataset = {}
    for class_name, tensors in classified_tensors.items():
        num_tensors = len(tensors)
        # single tensor, do 50-50 split
        if num_tensors == 2:
            # split both tensors 50-50 and combine for train and val/test
            tensor1 = tensors[0]
            tensor2 = tensors[1]
            train_tensor1, val_test_tensor1 = split_tensor(tensor1, 0.5, shuffle=True)
            train_tensor2, val_test_tensor2 = split_tensor(tensor2, 0.5, shuffle=True)
            train_tensor = torch.cat((train_tensor1, train_tensor2), dim=0)
            # split the val/test tensor 50-50
            val_tensor, test_tensor = split_tensor(torch.cat((val_test_tensor1, val_test_tensor2), dim=0), 0.5, shuffle=True)


        elif num_tensors == 4:
            # use first tensor to pick the used ligand for train/val
            used_ligand = tensors[0].ligand_name
            train_tensor1 = tensors[0]
            # find the tensor with the same ligand name for test
            train_tensor2 = None
            for t in tensors[1:]:
                if t.ligand_name == used_ligand:
                    train_tensor2 = t
                    break
            if train_tensor2 is None:
                raise ValueError(f"Could not find tensor with ligand {used_ligand} for test split.")

            # use the remaining for val/test
            remaining_tensors = [t for t in tensors if t.ligand_name != used_ligand]
            if len(remaining_tensors) != 2:
                raise ValueError(f"Expected 2 remaining tensors for test split, got {len(remaining_tensors)}.")
            val_test_tensor1 = remaining_tensors[0]
            val_test_tensor2 = remaining_tensors[1]
            val_tensor_1, test_tensor_1 = split_tensor(val_test_tensor1, 0.5, shuffle=True)
            val_tensor_2, test_tensor_2 = split_tensor(val_test_tensor2, 0.5, shuffle=True)
            val_tensor = torch.cat((val_tensor_1, val_tensor_2), dim=0)
            test_tensor = torch.cat((test_tensor_1, test_tensor_2), dim=0)
            del val_test_tensor1, val_test_tensor2
            train_tensor = torch.cat((train_tensor1, train_tensor2), dim=0)
            del train_tensor1, train_tensor2

        else:
            raise NotImplementedError

        train_dataset[class_name] = [train_tensor]
        val_dataset[class_name] = [val_tensor]
        test_dataset[class_name] = [test_tensor]

    return train_dataset, val_dataset, test_dataset


def create_splits_randomsplit(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Use PyTorch's random_split to split all frames randomly
    :param dataset:pytorch Dataset containing all trajectory frames and labels
    :param train_ratio: Ratio of training data
    :param val_ratio: Ratio of validation data
    :param test_ratio: Ratio of test data
    :param random_seed: Random seed for reproducibility
    :return: train_dataset, val_dataset, test_dataset - Dataset  containing the split data and labels
    """
    print('Creating splits...')
    torch.manual_seed(random_seed)

    # split the dataset
    datasets = random_split(dataset, [train_ratio, val_ratio, test_ratio])
    train_dataset, val_dataset, test_dataset = datasets



    return train_dataset, val_dataset, test_dataset


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

def save_image_datasets_to_folders(train_dataset, val_dataset, test_dataset, output_dir):
    """Save image datasets"""
    copy_dataset(train_dataset, os.path.join(output_dir, 'train'))
    copy_dataset(test_dataset, os.path.join(output_dir, 'test'))
    copy_dataset(val_dataset, os.path.join(output_dir, 'val'))

def train_val_test_split():
    """
    Split the data into train, validation and test sets
    :return:
    """
    config_parser  = ConfigParser("config.ini")

    for ds_size in ["small", "medium", "large"]:
        full_dataset_folder = f"./data/dataset/images/{ds_size}/full_dataset"
        dataset_folder = f"./data/dataset/images/{ds_size}/"

        train_size = float(config_parser.get("Train Validation Test Split", "Train Percent"))
        test_size = float(config_parser.get("Train Validation Test Split", "Validation Percent"))
        val_size = 1.0 - train_size - test_size
        # split the data into train, test and validation sets
        dataset = torchvision.datasets.ImageFolder(root=full_dataset_folder)
        train_dataset,val_dataset, test_dataset  = torch.utils.data.random_split(dataset, [train_size, val_size,test_size])

        # Save the datasets to their respective folders
        save_image_datasets_to_folders(train_dataset, val_dataset, test_dataset, output_dir=dataset_folder)

if __name__ == '__main__':
    train_val_test_split()