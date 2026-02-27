import torch
from fvcore.nn import FlopCountAnalysis, flop_count_str

from Models.custom_model_template import AbstractNNModel


class OneLayerNet(AbstractNNModel):
    def __init__(self, input_size, nr_neurons, output_size, dropout_rate=0.5):
        """
        Initialize the network architecture.

        Args:
            input_size: Number of input features (flattened image size)
            nr_neurons: Number of neurons in the hidden layer
            output_size: Number of output classes
            dropout_rate: Probability of dropping neurons (default 0.5 = 50%)
        """
        super(OneLayerNet, self).__init__()
        # First fully connected layer: input_size neurons -> nr_neurons neurons
        self.hidden_layer = torch.nn.Linear(input_size, nr_neurons)

        # Dropout layer: randomly disables neurons with probability dropout_rate
        self.dropout = torch.nn.Dropout(dropout_rate)
        # Output layer: nr_neurons -> output_size (number of classes)
        self.output_layer = torch.nn.Linear(nr_neurons, output_size)

    def forward(self, x):
        # Flatten input: convert 2D images to 1D vectors (keep batch dimension)
        x = x.view(x.size(0), -1)
        # Pass through hidden layer and apply ReLU activation
        x = torch.relu(self.hidden_layer(x))
        # Apply dropout for regularization during training
        x = self.output_layer(self.dropout(x))
        return x
    def input_shape(self):
        # Return the expected input shape for the model (excluding batch dimension)
        return 168 * 168 * 3,  # Assuming input images are 168x168 RGB (3 channels)




def create_onelayer_model():
    input_size = 168 * 168
    nr_neurons = 256
    output_size = 5
    model = OneLayerNet(input_size, nr_neurons, output_size)
    return model


