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

def load_model(model_class, model_filepath):
    model = model_class()
    model.load_state_dict(torch.load(model_filepath))
    return model

def validation_accuracy(model, validation_dataloader, device):
    model.eval()
    return Metrics.calculate_accuracy(model, validation_dataloader, device)




def training_loop(model, trainloader, testloader, optimizer, criterion, max_epochs=200):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == torch.device("cpu"):
        print("WARNING: Training on CPU, this may be slow. Consider using a GPU for faster training.")
    model.to(device)
    best_model = None

    metrics = Metrics(patience=10)
    for epoch in range(max_epochs):  # loop over the dataset multiple times
        model.train()
        running_loss = 0.0
        mini_batch_loss = 0.0
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

            running_loss += loss.item()
            mini_batch_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {mini_batch_loss / 2000:.3f}')
                mini_batch_loss = 0.0
        model.eval()
        metrics.update(trainloader=trainloader, testloader=testloader, model=model, running_loss=running_loss, criterion=criterion, device=device)


        print \
            (f'Epoch {epoch + 1} - Training Accuracy: {metrics.training_accuracy[-1]:.2f}%, Test Accuracy: {metrics.test_accuracy[-1]:.2f}%')

        if metrics.is_overfitting():
            print("Early stopping due to overfitting.")
            return best_model, metrics
        else:
            best_model = model.state_dict()

    print('Finished Training')
    return best_model, metrics



