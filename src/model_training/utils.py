import os
import time
import datasets

import torch

from src.model_training.Metrics import Metrics
from src.utils.configParser import ConfigParser
import matplotlib.pyplot as plt

def model_name(model_name_prefix):
    # current date and time
    date = time.strftime("%Y%m%d-%H%M%S")
    model_name = f"{model_name_prefix}_{date}"
    return model_name
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("WARNING: Training on CPU, this may be slow. Consider using a GPU for faster training.")
    return device
def load_model(model_class, model_filepath):
    model = model_class()
    model.load_state_dict(torch.load(model_filepath))
    return model
def get_subset(dataset, fraction, shuffle=True):
    if fraction <= 0 or fraction > 1:
        raise ValueError("Fraction must be between 0 and 1.")
    dataset_size = int(max(len(dataset) * fraction, 1))
    if shuffle:
        dataset = dataset.shuffle()
    return dataset.select(range(dataset_size))
def encode_labels(labels):
    label_mapping = {
        "nonbinder": 0,
        "dpp9selective": 1,
        "dpp8selective": 2,
        "aselective": 3,
        "apo": 4
    }
    labels = [label_mapping[label] for label in labels]
    return labels

def train_val_test_split(dataset, train_fraction=0.7, val_fraction=0.15, seed=42):
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("Train and validation fractions must sum to less than 1.0")

    # First split into train and temp (val + test)
    temp_fraction = 1.0 - train_fraction
    train_val_test = dataset.train_test_split(test_size=temp_fraction, seed=seed, shuffle=True)
    train_set = train_val_test['train']
    temp_set = train_val_test['test']

    # Now split temp into validation and test
    test_fraction = 1.0 - (val_fraction / temp_fraction)
    val_test = temp_set.train_test_split(test_size=test_fraction, seed=seed, shuffle=True)
    val_set = val_test['train']
    test_set = val_test['test']

    return {"train": train_set, "val": val_set,"test": test_set}

def training_loop(model, trainloader, validationloader, optimizer, criterion, max_epochs=200):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == torch.device("cpu"):
        print("WARNING: Training on CPU, this may be slow. Consider using a GPU for faster training.")
    model.to(device)
    best_model_state_dict = None
    epochs_best_model = None
    metrics = Metrics(patience=10)
    for epoch in range(max_epochs):  # loop over the dataset multiple times
        model.train()
        running_loss = 0.0
        mini_batch_loss = 0.0
        for i, batch in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = batch["data"], batch["labels"]
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
        metrics.update(trainloader=trainloader, validationloader=validationloader, model=model, running_loss=running_loss, criterion=criterion, device=device)


        print \
            (f'Epoch {epoch + 1} - Training Accuracy: {metrics.training_accuracy[-1]:.4f}%, Validation Accuracy: {metrics.validation_accuracy[-1]:.4f}%')

        if metrics.is_overfitting():
            print("Early stopping due to overfitting.")
            return best_model_state_dict, epochs_best_model,metrics
        else:
            best_model_state_dict = model.state_dict()
            epochs_best_model = epoch

    print('Finished Training')
    return best_model_state_dict, epochs_best_model,metrics



