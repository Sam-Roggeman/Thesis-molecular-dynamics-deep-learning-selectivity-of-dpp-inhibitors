from pprint import pprint

import torchvision

from src.model_training.utils import save_model, training_loop
from src.utils.configParser import ConfigParser
import time
import os
import torch.optim as optim

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import torch.nn as nn
import torch.nn.functional as F
from src.utils.DataLoader import load_dataset, load_dataset_from_config
from src.utils.Metrics import Metrics



class SimpleCNN(nn.Module):
    def __init__(self, input_size=166):
        super().__init__()


        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # Calculate the flattened size
        # After conv1: (input_size - 4), after pool: (input_size - 4) // 2
        # After conv2: (input_size - 4) // 2 - 4, after pool: ((input_size - 4) // 2 - 4) // 2
        conv_output_size = ((input_size - 4) // 2 - 4) // 2
        flattened_size = 16 * conv_output_size * conv_output_size

        self.fc1 = nn.Linear(flattened_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



def train_model():
    simple_cnn = SimpleCNN()
    config_parser = ConfigParser("config.ini")
    trainloader, validationloader, _ = load_dataset_from_config()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(simple_cnn.parameters(), lr=0.001, momentum=0.9)
    model, metrics = training_loop(model=simple_cnn, trainloader=trainloader, validationloader=validationloader,
                                   optimizer=optimizer, criterion=criterion)
    model_prefix = "SimpleCNN"
    _, filename = save_model(simple_cnn, model_prefix)
    metrics.plot_metrics(filename)

if __name__ == "__main__":
    train_model()

