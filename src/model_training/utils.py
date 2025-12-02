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
    if 0.9999 < fraction <= 1.0:
        return dataset
    if fraction <= 0 or fraction > 1.0001:
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

def training_loop(model, trainloader, validationloader, optimizer, criterion, model_folder,max_epochs=200, patience=10, time_limit=None ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric_path =os.path.join(model_folder, f'metrics_training_loop.pt')
    plot_path = os.path.join(model_folder, f'plots_training_loop.png')
    print(f"Using device: {device}")
    print(f"Using metrics path: {metric_path}")
    print(f"Using patience: {patience}")
    # size of trainloader and validationloader
    print(f"Training set size: {len(trainloader.dataset)}")
    print(f"Validation set size: {len(validationloader.dataset)}")
    # time limit in readable format for logging
    if time_limit:
        time_limit_str = f"{time_limit // 3600}h {(time_limit % 3600) // 60}m {time_limit % 60}s"
        print(f"Using time limit: {time_limit_str}.")
    print(f"Using plot path: {plot_path}")

    if device == torch.device("cpu"):
        print("WARNING: Training on CPU, this may be slow. Consider using a GPU for faster training.")
    model.to(device)
    best_model_state_dict = None
    epochs_best_model = None
    metrics = Metrics(patience=patience)
    start_time = time.time()
    for epoch in range(max_epochs):  # loop over the dataset multiple times
        model.train()
        correct = 0
        total = 0
        running_loss = 0.0
        start_time_epoch = time.time()
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

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            running_loss += loss.item()
        train_loss = running_loss / len(trainloader)
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0


        with torch.no_grad():
            for i, batch in enumerate(validationloader, 0):
                inputs, labels = batch["data"], batch["labels"]
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(validationloader)
        val_acc = correct / total
        metrics.update(train_acc * 100, train_loss, val_acc * 100, val_loss)
        # override metrics and plot
        metrics.save_metrics(metric_path)
        metrics.save_plot("Training and Validation Metrics", plot_path)


        print(f'Epoch {epoch + 1}:')
        average_time_per_epoch = (time.time() - start_time) / (epoch + 1)
        time_epoch = time.time() - start_time_epoch
        print(f'\tTraining  \tAccuracy: {metrics.training_accuracy[-1]:.4f}%\tLoss: {metrics.train_loss[-1]:.4f}')
        print(f'\tValidation\tAccuracy: {metrics.validation_accuracy[-1]:.4f}%\tLoss: {metrics.validation_loss[-1]:.4f}')
        print('-'*100)
        # print patience counter
        if metrics.is_overfitting():
            print("Early stopping due to overfitting.")
            return best_model_state_dict, epochs_best_model,metrics
        elif time_limit and (time.time() - start_time) > time_limit:
            print("Time limit reached, stopping training.")
            return best_model_state_dict, epochs_best_model,metrics
        elif metrics.model_improved():
            best_model_state_dict = model.state_dict()
            # save the best model
            path = os.path.join(model_folder, f'current_best_model.pth')
            torch.save(best_model_state_dict, path)
            print(f"\tSaved best model at epoch {epoch+1} to {path}")

            epochs_best_model = epoch
        print(f'\tPatience Counter: {metrics.patience_counter}/{patience}')
        print(f'\tTime for epoch:   {time_epoch//60:.2f}m {time_epoch%60:.0f}s\t(avg: {average_time_per_epoch//60}m {average_time_per_epoch%60:.0f}s/epoch)\t Time elapsed since start: {(time.time() - start_time)//60:.0f}m')
        print('='*100)

    print('Finished Training')
    return best_model_state_dict, epochs_best_model,metrics



