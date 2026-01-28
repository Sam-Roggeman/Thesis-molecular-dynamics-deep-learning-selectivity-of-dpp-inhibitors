import torch


class OneLayerNet(torch.nn.Module):
    def __init__(self, input_size, nr_neurons, output_size, dropout_rate=0.5):
        super(OneLayerNet, self).__init__()
        # hidden layer
        self.hidden_layer = torch.nn.Linear(input_size, nr_neurons)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.output_layer = torch.nn.Linear(nr_neurons, output_size)

    def forward(self, x):
        return self.output_layer(self.dropout(torch.relu(self.hidden_layer(x.view(x.size(0), -1)))))

def create_onelayer_model():
    input_size = 168 * 168
    nr_neurons = 256
    output_size = 5
    model = OneLayerNet(input_size, nr_neurons, output_size)
    return model
