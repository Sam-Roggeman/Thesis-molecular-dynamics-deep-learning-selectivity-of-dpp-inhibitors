import os
import datasets
import numpy as np
import torchvision

from src.Transform.ListScrambler import ListScrambler, ScramblingTransform
from src.Transform.Padder import Padder
from src.Transform.XYZToRGBTensor import XYZToRGBTensor
from src.model_training.metric_functions import calculate_accuracy
from src.model_training.utils import training_loop, load_model, encode_labels, model_name, get_device, get_subset
import torch.optim as optim
from src.model_training.DataLoader import load_dataset_from_safetensors_multichunk, \
    load_validation_from_safetensors_multichunk
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.configParser import ConfigParser
from src.utils.logger import replace_output, setup_logger


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size=4):
        super(_DenseLayer, self).__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate,
                               kernel_size=1, stride=1, bias=False)

        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate,
                               kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        # Bottleneck layer
        out = self.relu1(self.conv1(self.norm1(x)))
        out = self.relu2(self.conv2(self.norm2(out)))
        out = torch.cat([x, out], 1)
        return out


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, growth_rate, bn_size=4):
        super(_DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size
            )
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class _Transition(nn.Module):
    def __init__(self, num_input_features, reduction_ratio=0.5):
        super(_Transition, self).__init__()
        num_output_features = int(num_input_features * reduction_ratio)
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features,
                              kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv(self.relu(self.norm(x)))
        out = self.pool(out)
        return out


class CustomDenseNet(nn.Module):
    def __init__(self, growth_rate=48, block_config=(6, 12, 36, 24),
                 num_init_features=96, reduction_ratio=0.5, num_classes=1000, bn_size=4):
        super(CustomDenseNet, self).__init__()

        # Initial convolution
        self.features = nn.Sequential(
            nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Dense blocks
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                growth_rate=growth_rate,
                bn_size=bn_size
            )
            self.features.add_module(f'denseblock{i + 1}', block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    reduction_ratio=reduction_ratio)
                self.features.add_module(f'transition{i + 1}', trans)
                num_features = int(num_features * reduction_ratio)

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Classifier
        self.classifier = nn.Linear(num_features, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


# Create the model with your specified parameters
def create_custom_densenet(num_classes=1000):
    model = CustomDenseNet(
        growth_rate=48,
        block_config=(6, 12, 36, 24),  # 4 dense blocks with 6, 12, 36, 24 layers
        num_init_features=96,  # 96 initial filters
        reduction_ratio=0.5,  # reduction ratio of 0.5
        num_classes=num_classes
    )
    return model


def train_model():
    cdcnn = CustomDenseNet()
    config_parser = ConfigParser("config.ini")

    log_dir = "./logs/full_dataset_training"
    logger = setup_logger(log_file="", log_dir=log_dir, logging_enabled=True,
                          console_enabled=False)
    replace_output(logger)

    dataset_folder = "./data/dataset/full_dataset/"

    device = get_device()

    dataset = datasets.load_from_disk(dataset_folder)

    # Load only dataset_size% of each dataset
    dataset_size = 0.25
    dataset = get_subset(dataset, dataset_size)


    # rename coords = data and binding_type = labels
    data_col = "coordinates"
    label_col = "binding_type"

    dataset = dataset.rename_column(data_col, "data").rename_column(label_col, "labels")

    split_ratio = [0.7, 0.15, 0.15]
    dataset = dataset.train_test_split(test_size=split_ratio[2] + split_ratio[1], seed=42, shuffle=True)
    trainset = dataset['train']
    val_test_set = dataset['test']
    val_test_set = val_test_set.train_test_split(test_size=split_ratio[2]/(split_ratio[1]+split_ratio[2]), seed=42, shuffle=True)
    valset = val_test_set['train']
    testset = val_test_set['test']
    # save the test set for later evaluation
    testset.save_to_disk(os.path.join(dataset_folder, "test_set"))

    def apply_transform(examples_data, examples_labels, real_nr_atoms):
        """Apply transform to each entry in the batch"""
        # Change from XYZ (tensor shape: [28224,3]) to RGB [3,168,168]
        rgb_transformer = XYZToRGBTensor(target_size=168)
        # Scramble with diameter 140A
        scrambler = ScramblingTransform(140)
        # Pad to 168x168 = 28224
        padder = Padder(target_size=168, fill=0)

        examples_data = rgb_transformer(padder(scrambler(examples_data)), real_nr_atoms)
        examples_labels = encode_labels(examples_labels)

        return {"data": examples_data, "labels": examples_labels}
    def apply_transform_val(examples_data, examples_labels, real_nr_atoms):
        """Apply transform to each entry in the batch"""
        # Change from XYZ (tensor shape: [28224,3]) to RGB [3,168,168]
        rgb_transformer = XYZToRGBTensor(target_size=168)
        # Pad to 168x168 = 28224
        padder = Padder(target_size=168, fill=0)

        examples_data = rgb_transformer(padder(examples_data), real_nr_atoms)
        examples_labels = encode_labels(examples_labels)

        return {"data": examples_data, "labels": examples_labels}
    trainset = trainset.map(
        apply_transform,
        batch_size=128,
        batched=True,
        input_columns=['data', 'labels', "num_atoms"],
        remove_columns=['pdb_id', 'dpp_class', 'ligand_name', 'num_atoms'],
        num_proc=8
    )

    valset = valset.map(
        apply_transform_val,
        batch_size=128,
        batched=True,
        input_columns=['data', 'labels',"num_atoms"],
        remove_columns=['pdb_id', 'dpp_class', 'ligand_name', 'num_atoms'],
        num_proc=8

    )

    trainset.set_format(type='torch', columns=['data', 'labels'], device=device)
    valset.set_format(type='torch', columns=['data', 'labels'], device=device)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
    validationloader = torch.utils.data.DataLoader(valset, batch_size=32, shuffle=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(cdcnn.parameters(), lr=0.001, momentum=0.9)

    model_state_dict, nr_epochs, metrics  = training_loop(model=cdcnn,
                                                         trainloader=trainloader,
                                                         validationloader=validationloader,
                                                         optimizer=optimizer,
                                                         criterion=criterion,
                                                         max_epochs=200, )
    model = create_custom_densenet()
    model.load_state_dict(model_state_dict)

    model_prefix = f"DenseCNN_full_dataset_{nr_epochs}epochs"
    path = str(os.path.join(config_parser.get("Model Training", "Model Save Folder")))
    subdir = model_name(model_prefix)
    path = os.path.join(path, subdir)
    os.makedirs(path, exist_ok=False)
    filename = f"{model_prefix}_{model_prefix}.pth"
    torch.save(model_state_dict, os.path.join(path, filename))

    print(f"Model saved to: {os.path.join(path, filename)}")
    plot_path = filename.replace('.pth', '.png')
    metric_path = filename.replace('.pth', '.metrics')
    metrics.save_plot( "DCNN on the ligand split data" ,os.path.join(path, plot_path))
    metrics.save_metrics(os.path.join(path, metric_path))

    print(f"Plot saved to: {os.path.join(path, plot_path)}")
    test_acc = calculate_accuracy(model, test_loader, device=device)
    print(f"Test Accuracy: {test_acc}")

def train_model_ligand_split():
    cdcnn = CustomDenseNet()
    config_parser = ConfigParser("config.ini")

    log_dir = "./logs/Ligand_Split_Training"
    logger = setup_logger(log_file="", log_dir=log_dir, logging_enabled=True,
                          console_enabled=False)
    replace_output(logger)

    dataset_folder = "./data/dataset/ligand_set/"

    device = get_device()

    trainset = datasets.load_from_disk(os.path.join(dataset_folder, "ligand_split_train"))
    valset = datasets.load_from_disk(os.path.join(dataset_folder, "ligand_split_val"))

    # Load only dataset_size% of each dataset
    dataset_size = 0.25
    trainset = get_subset(trainset, dataset_size)
    valset = get_subset(valset, dataset_size)


    # rename coords = data and binding_type = labels
    data_col = "coordinates"
    label_col = "binding_type"

    trainset = trainset.rename_column(data_col, "data")
    valset = valset.rename_column(data_col, "data")
    trainset = trainset.rename_column(label_col, "labels")
    valset = valset.rename_column(label_col, "labels")

    def apply_transform(examples_data, examples_labels, real_nr_atoms):
        """Apply transform to each entry in the batch"""
        # Change from XYZ (tensor shape: [28224,3]) to RGB [3,168,168]
        rgb_transformer = XYZToRGBTensor(target_size=168)
        # Scramble with diameter 140A
        scrambler = ScramblingTransform(140)
        # Pad to 168x168 = 28224
        padder = Padder(target_size=168, fill=0)

        examples_data = rgb_transformer(padder(scrambler(examples_data)), real_nr_atoms)
        examples_labels = encode_labels(examples_labels)

        return {"data": examples_data, "labels": examples_labels}

    trainset = trainset.map(
        apply_transform,
        batch_size=128,
        batched=True,
        input_columns=['data', 'labels', "num_atoms"],
        remove_columns=['pdb_id', 'dpp_class', 'ligand_name', 'num_atoms'],
        num_proc=8
    )

    valset = valset.map(
        apply_transform,
        batch_size=128,
        batched=True,
        input_columns=['data', 'labels',"num_atoms"],
        remove_columns=['pdb_id', 'dpp_class', 'ligand_name', 'num_atoms'],
        num_proc=8

    )

    trainset.set_format(type='torch', columns=['data', 'labels'], device=device)
    valset.set_format(type='torch', columns=['data', 'labels'], device=device)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
    validationloader = torch.utils.data.DataLoader(valset, batch_size=32, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(cdcnn.parameters(), lr=0.001, momentum=0.9)

    model_state_dict, nr_epochs, metrics  = training_loop(model=cdcnn,
                                                         trainloader=trainloader,
                                                         validationloader=validationloader,
                                                         optimizer=optimizer,
                                                         criterion=criterion,
                                                         max_epochs=200, )
    model_prefix = f"DenseCNN_ligand_split_{nr_epochs}epochs"
    path = str(os.path.join(config_parser.get("Model Training", "Model Save Folder")))
    subdir = model_name(model_prefix)
    path = os.path.join(path, subdir)
    os.makedirs(path, exist_ok=False)
    filename = f"{model_prefix}_{model_prefix}.pth"
    torch.save(model_state_dict, os.path.join(path, filename))

    print(f"Model saved to: {os.path.join(path, filename)}")
    plot_path = filename.replace('.pth', '.png')
    metric_path = filename.replace('.pth', '.metrics')
    metrics.save_plot( "DCNN on the ligand split data" ,os.path.join(path, plot_path))
    metrics.save_metrics(os.path.join(path, metric_path))

    print(f"Plot saved to: {os.path.join(path, plot_path)}")



def validate_model():
    config_parser = ConfigParser("config.ini")
    dataset_folder = config_parser.get("Model Training", "Input Folder")
    validation_loader = load_validation_from_safetensors_multichunk(dataset_folder)
    model_path = config_parser.get("Model Training", "Model Save Folder")
    model_filename = "DenseCNN_51epochs_20251102-155816.pth"
    full_model_path = os.path.join(model_path, model_filename)
    model = load_model(model_class=CustomDenseNet, model_filepath=full_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_acc = calculate_accuracy(model, validation_loader, device=device)
    print(f"Validation Accuracy: {val_acc}")


if __name__ == "__main__":
    train_model()
