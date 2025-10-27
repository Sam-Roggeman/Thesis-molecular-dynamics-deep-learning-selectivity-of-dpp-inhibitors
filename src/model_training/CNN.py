import torch
import torchvision
import torchvision.transforms as transforms
from src.utils.configParser import ConfigParser
import time
import os
import shutil
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
def copy_dataset(dataset, split_name):
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    for idx in tqdm(range(len(dataset)), desc=f"Copying {split_name}"):
        # Get the original file path from dataset
        original_idx = dataset.indices[idx]
        original_path, label = dataset.dataset.samples[original_idx]

        # Get class name from path
        class_name = os.path.basename(os.path.dirname(original_path))

        # Create class directory
        class_dir = os.path.join(split_dir, class_name)
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
    all_data_folder = config_parser.get("Data Embedding", "folder")

    dataset_folder = config_parser.get("Data Embedding", "dataset folder")
    # split the data into train, test and validation sets
    dataset = torchvision.datasets.ImageFolder(root=all_data_folder)
    train_size = int(0.7 * len(dataset))
    test_size = int(0.15 * len(dataset))
    val_size = len(dataset) - train_size - test_size

    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size, val_size])

    # Save the datasets to their respective folders
    save_image_datasets_to_folders(train_dataset, test_dataset, val_dataset, output_dir=dataset_folder)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_model():
    config_parser  = ConfigParser("config.ini")
    dataset_folder = config_parser.get("Data Embedding", "dataset folder")
    train_folder = os.path.join(dataset_folder, 'train')
    test_folder = os.path.join(dataset_folder, 'test')
    trainset = torchvision.datasets.ImageFolder(root=train_folder)
    testset = torchvision.datasets.ImageFolder(root=test_folder)

    batch_size = 4

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

    save_folder = config_parser.get("Data Embedding", "Model Save Folder")
    os.makedirs(save_folder, exist_ok=True)
    # current date and time
    date = time.strftime("%Y%m%d-%H%M%S")
    model_name = f"cnn_model_{date}"
    model_filename = f"{model_name}.pth"
    PATH = os.path.join(save_folder, model_filename)
    torch.save(net.state_dict(), PATH)
    print(f"Model saved to {PATH}")

