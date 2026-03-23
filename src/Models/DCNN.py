import os
from datetime import datetime

import datasets
import numpy as np
import torchvision

from src.Models.custom_model_template import AbstractNNModel
from src.Transform.ListScrambler import ListScrambler, ScramblingTransform
from src.Transform.Padder import Padder
from src.Transform.XYZToRGBTensor import XYZToRGBTensor
from src.model_training.utils import training_loop, load_model, model_name, get_device, get_subset
from src.model_training.LabelEncoder import LabelEncoder
import torch.optim as optim
from src.model_training.DataLoader import load_dataset_from_safetensors_multichunk, \
    load_validation_from_safetensors_multichunk
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.Transform.tranformators import apply_image_transform, apply_image_transform_noscramble
from src.utils.configParser import ConfigParser
from src.utils.logger import replace_output, setup_logger
from src.utils.training_config import TrainingConfig
from src.utils.training_setup import train_model

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size=4, dropout_rate=0.2):
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


class CustomDenseNet(AbstractNNModel):
    def __init__(self, growth_rate=48, block_config=(6, 12, 36, 24),
                 num_init_features=96, reduction_ratio=0.5, num_classes=5, bn_size=4, dropout_rate=0.5):
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
        self.dropout = nn.Dropout(dropout_rate)
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
        out = self.dropout(out)
        out = self.classifier(out)
        return out

    def input_shape(self):
        return 3, 168, 168  # Assuming input images are 168x168 RGB (3 channels)


# Create the model with your specified parameters
def create_custom_densenet(num_classes=5):
    model = CustomDenseNet(
        growth_rate=48,
        block_config=(6, 12, 36, 24),  # 4 dense blocks with 6, 12, 36, 24 layers
        num_init_features=96,  # 96 initial filters
        reduction_ratio=0.5,  # reduction ratio of 0.5
        num_classes=num_classes
    )
    return model

