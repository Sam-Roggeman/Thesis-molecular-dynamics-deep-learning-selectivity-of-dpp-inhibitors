import os
import time

import torch

from src.utils.Metrics import Metrics
from src.utils.configParser import ConfigParser
import matplotlib.pyplot as plt


def save_model(model_state_dict, model_name_prefix):
    config_parser = ConfigParser("config.ini")
    save_folder = config_parser.get("Model Training", "Model Save Folder")
    os.makedirs(save_folder, exist_ok=True)
    # current date and time
    date = time.strftime("%Y%m%d-%H%M%S")
    model_name = f"{model_name_prefix}_{date}"
    model_filename = f"{model_name}.pth"
    PATH = os.path.join(save_folder, model_filename)
    torch.save(model_state_dict, PATH)
    return PATH, model_filename


def training_loop(model, trainloader, testloader, optimizer, criterion ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    best_model = None

    metrics = Metrics(patience=10)
    for epoch in range(200):  # loop over the dataset multiple times
        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
        model.eval()
        metrics.update(trainloader, testloader, model, running_loss, device)
        val_loss = metrics.last_val_loss()

        print \
            (f'Epoch {epoch + 1} - Training Accuracy: {metrics.training_accuracy[-1]:.2f}%, Test Accuracy: {metrics.test_accuracy[-1]:.2f}%')

        if metrics.is_overfitting():
            print("Early stopping due to overfitting.")
            return best_model, metrics
        else:
            best_model = model.state_dict()

    print('Finished Training')
    return best_model, metrics

def save_tensors(tensors, file_path):
    """
    save tensors using .pt file extension
    """
    torch.save(tensors, file_path)

