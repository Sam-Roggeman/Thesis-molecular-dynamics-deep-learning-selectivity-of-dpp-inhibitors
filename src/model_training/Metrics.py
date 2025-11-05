import torch
import matplotlib.pyplot as plt


class Metrics:
    def __init__(self, patience=5):
        self.test_loss = []
        self.test_accuracy = []
        self.training_accuracy = []
        self.train_loss = []
        self.min_val_loss = float('inf')
        self.patience = patience
        self.patience_counter = 0

    @staticmethod
    def calculate_accuracy(model, dataloader, device):
        correct = 0
        total = 0

        with torch.no_grad():
            for data in dataloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        return accuracy

    @staticmethod
    def calculate_loss(model, dataloader, criterion, device):
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in dataloader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(dataloader)
        return val_loss

    def update(self, trainloader, testloader, model, running_loss, criterion, device):
        self.train_loss.append(running_loss / len(trainloader))

        # accuracy on training set
        train_accuracy = self.calculate_accuracy(model, trainloader, device)
        # accuracy on test set
        test_accuracy = self.calculate_accuracy(model, testloader, device)
        test_loss = self.calculate_loss(model, testloader, criterion, device)


        self.training_accuracy.append(train_accuracy)
        self.test_accuracy.append(test_accuracy)
        self.test_loss.append(test_loss)


    def to_dict(self):
        return {
            "test_accuracy": self.test_accuracy,
            "training_accuracy": self.training_accuracy,
            "loss": self.train_loss
        }

    def nr_epochs(self):
        return len(self.training_accuracy)

    def plot_metrics(self, title, ):
        """
        :param metrics: dict: {"accuracy":[values], "loss":[values]}
        :return:
        """
        test_color = 'b'
        train_color = 'r'

        epochs = range(1, len(self.training_accuracy) + 1)
        plt.figure(figsize=(12, 5))
        plt.title(title)
        # disable y and x axis for the main plot
        plt.axis('off')

        # accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.test_accuracy, test_color, label='Test Accuracy')
        plt.plot(epochs, self.training_accuracy, train_color, label='Train Accuracy')
        plt.title('Test Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        # loss plot
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.test_loss, test_color, label='Test Loss')
        plt.plot(epochs, self.train_loss, train_color, label='Train Loss')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()
        # return plot such that it can be saved externally
        return plt






    def is_overfitting(self):
        # if not enough data to compare return False
        if len(self.training_accuracy) < 2 or len(self.test_accuracy) < 2:
            return False
        val_loss_last = self.test_loss[-1]
        if val_loss_last < self.min_val_loss:
            self.min_val_loss = val_loss_last
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                return True
        return False