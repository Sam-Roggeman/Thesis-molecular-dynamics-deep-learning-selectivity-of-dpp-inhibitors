import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.custom_model_template import AbstractNNModel


class SimpleCNN(AbstractNNModel):
    def __init__(self, input_size=168, dropout_rate=0.5):
        super().__init__()

        # First convolutional layer: 3 input channels (RGB) -> 6 output filters, kernel size 5x5
        self.conv1 = nn.Conv2d(3, 6, 5)
        # Max pooling: reduces spatial dimensions by factor of 2
        self.pool = nn.MaxPool2d(2, 2)
        # Second convolutional layer: 6 input channels -> 16 output filters, kernel size 5x5
        self.conv2 = nn.Conv2d(6, 16, 5)

        # Calculate the flattened size
        # Each 5x5 conv reduces dimensions by 4 (for valid padding)
        # Each 2x2 pooling divides dimensions by 2
        # After conv1: (input_size - 4), after pool: (input_size - 4) // 2
        # After conv2: (input_size - 4) // 2 - 4, after pool: ((input_size - 4) // 2 - 4) // 2
        conv_output_size = ((input_size - 4) // 2 - 4) // 2

        # Total flattened size = 16 filters * height * width
        flattened_size = 16 * conv_output_size * conv_output_size

        # First dense layer: flattened conv output -> 120 neurons
        self.fc1 = nn.Linear(flattened_size, 120)

        # Second dense layer: 120 neurons -> 84 neurons
        self.fc2 = nn.Linear(120, 84)

        # Dropout layer: randomly zeros out neurons with probability dropout_rate
        self.dropout = nn.Dropout(p=dropout_rate)

        # Output layer: 84 neurons -> 5 classes
        self.fc3 = nn.Linear(84, 5)

    def forward(self, x):
        # First conv block: convolution -> ReLU activation -> max pooling
        x = self.pool(F.relu(self.conv1(x)))
        # Second conv block: convolution -> ReLU activation -> max pooling
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten all dimensions except the batch dimension
        x = torch.flatten(x, 1)
        # First fully connected layer with ReLU activation
        x = F.relu(self.fc1(x))
        # Second fully connected layer with ReLU activation
        x = F.relu(self.fc2(x))
        # Apply dropout for regularization during training
        x = self.dropout(x)
        # Output layer (no activation - raw logits for classification)
        x = self.fc3(x)
        return x
    def input_shape(self):
        # Return the expected input shape for the model (excluding batch dimension)
        return 3, 168, 168  # Assuming input images are 168x168 RGB (3 channels)