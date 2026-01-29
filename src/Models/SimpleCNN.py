import torch
import torch.nn as nn
import torch.nn.functional as F




class SimpleCNN(nn.Module):
    def __init__(self, input_size=168, dropout_rate=0.5):
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
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc3 = nn.Linear(84, 5)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x




