import torch


class OneLayerNet(torch.nn.Module):
    def __init__(self, input_size, nr_neurons, output_size):
        super(OneLayerNet, self).__init__()
        # hidden layer
        self.hidden_layer = torch.nn.Linear(input_size, nr_neurons)
        self.output_layer = torch.nn.Linear(nr_neurons, output_size)

    def forward(self, x):
        return self.output_layer(torch.relu(self.hidden_layer(x)))

def create_onelayer_model():
    input_size = 168 * 168
    nr_neurons = 256
    output_size = 5
    model = OneLayerNet(input_size, nr_neurons, output_size)
    return model

def train_model():
    onelayer_model = create_onelayer_model()
    dataset_folder = os.path.join()

    dataset_folder = config_parser.get("Model Training", "Input Folder")
    trainloader, validationloader, _ = load_dataset_from_safetensors_multichunk(dataset_folder)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(cdcnn.parameters(), lr=0.001, momentum=0.9)
    model_state_dict, metrics = training_loop(model=cdcnn,
                                              trainloader=trainloader,
                                              validationloader=validationloader,
                                              optimizer=optimizer,
                                              criterion=criterion,
                                              max_epochs=200)
    nr_epochs = metrics.nr_epochs()
    model_prefix = f"DenseCNN_{nr_epochs}epochs"
    _, filename = save_model(model_state_dict, model_prefix)
    metrics.save_plot(filename)