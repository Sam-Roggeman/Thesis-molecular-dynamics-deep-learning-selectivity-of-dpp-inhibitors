import torchvision
import torch
import os

from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms

from src.utils.configParser import ConfigParser


def load_dataset(dataset_folder):
    train_folder = os.path.join(dataset_folder, 'train')
    test_folder = os.path.join(dataset_folder, 'test')
    validate_folder = os.path.join(dataset_folder, 'val')

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.ImageFolder(root=train_folder, transform=transform)
    testset = torchvision.datasets.ImageFolder(root=test_folder, transform=transform)
    validateset = torchvision.datasets.ImageFolder(root=validate_folder, transform=transform)

    batch_size = 16

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)
    validateloader = torch.utils.data.DataLoader(validateset, batch_size=batch_size,
                                                 shuffle=False, num_workers=8)

    return trainloader, testloader, validateloader


def save_images_as_tensors(dataset_folder, output_file='dataset.pt'):
    """Read all PNGs once and save as a single tensor file"""
    train_folder = os.path.join(dataset_folder, 'train')
    test_folder = os.path.join(dataset_folder, 'test')
    validate_folder = os.path.join(dataset_folder, 'val')

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    print("Loading images from disk...")
    trainset = torchvision.datasets.ImageFolder(root=train_folder, transform=transform)
    testset = torchvision.datasets.ImageFolder(root=test_folder, transform=transform)
    validateset = torchvision.datasets.ImageFolder(root=validate_folder, transform=transform)

    # Convert to tensors
    print("Converting to tensors...")
    train_images = torch.stack([img for img, _ in trainset])
    train_labels = torch.tensor([label for _, label in trainset])

    test_images = torch.stack([img for img, _ in testset])
    test_labels = torch.tensor([label for _, label in testset])

    val_images = torch.stack([img for img, _ in validateset])
    val_labels = torch.tensor([label for _, label in validateset])

    # Save everything to one file
    print(f"Saving to {output_file}...")
    torch.save({
        'train_images': train_images,
        'train_labels': train_labels,
        'test_images': test_images,
        'test_labels': test_labels,
        'val_images': val_images,
        'val_labels': val_labels,
        'classes': trainset.classes
    }, output_file)

    print(f"Done! Saved {len(train_images)} train, {len(test_images)} test, {len(val_images)} val images")


def load_dataset_from_tensors(tensor_file='dataset.pt', batch_size=16):
    """Load the pre-saved tensor file"""
    print(f"Loading tensors from {tensor_file}...")
    data = torch.load(tensor_file)

    # Create TensorDatasets
    trainset = TensorDataset(data['train_images'], data['train_labels'])
    testset = TensorDataset(data['test_images'], data['test_labels'])
    validateset = TensorDataset(data['val_images'], data['val_labels'])

    # Create DataLoaders
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    validateloader = DataLoader(validateset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Loaded! Train: {len(trainset)}, Test: {len(testset)}, Val: {len(validateset)}")

    return trainloader, testloader, validateloader

if __name__ == '__main__':
    config_parser = ConfigParser("config.ini")
    # dataset_folder = config_parser.get("Data Loader", "Dataset Folder")
    # output_tensor_file = config_parser.get("Data Loader", "Output Tensor File")
    for s in ["large"]:
        save_images_as_tensors(f"./data/dataset/images/{s}", f"./data/dataset/images/{s}.pt")

