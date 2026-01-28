import torch
import matplotlib.pyplot as plt
from src.model_training.metric_functions import calculate_accuracy, calculate_loss


class Metrics:
    def __init__(self, patience=5):
        self.validation_loss = []
        self.validation_accuracy = []
        self.training_accuracy = []
        self.train_loss = []
        self.min_val_loss = float('inf')
        self.patience = patience
        self.patience_counter = 0


    def update(self, train_acc, train_loss, val_acc, val_loss):
        self.train_loss.append(train_loss)
        self.training_accuracy.append(train_acc)
        self.validation_accuracy.append(val_acc)
        self.validation_loss.append(val_loss)


    def to_dict(self):
        return {
            "validation_accuracy": self.validation_accuracy,
            "training_accuracy": self.training_accuracy,
            "loss": self.train_loss
        }

    def nr_epochs(self):
        return len(self.training_accuracy)
    def save_metrics(self, filename):
        metrics_dict = self.to_dict()
        torch.save(metrics_dict, filename)
    def load_metrics(self, filename):
        metrics_dict = torch.load(filename)
        self.validation_accuracy = metrics_dict.get("validation_accuracy", [])
        self.training_accuracy = metrics_dict.get("training_accuracy", [])
        self.train_loss = metrics_dict.get("loss", [])



    def save_plot(self, title, filename):
        """
        :param metrics: dict: {"accuracy":[values], "loss":[values]}
        :return:
        """
        validation_color = 'b'
        train_color = 'r'

        epochs = range(1, len(self.training_accuracy) + 1)
        plt.figure(figsize=(12, 5))
        plt.title(title, pad=20)

        # disable y and x axis for the main plot
        plt.axis('off')

        # accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.validation_accuracy, validation_color, label='validation Accuracy')
        plt.plot(epochs, self.training_accuracy, train_color, label='Train Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('% Accuracy')
        plt.legend()

        # loss plot
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.validation_loss, validation_color, label='validation Loss')
        plt.plot(epochs, self.train_loss, train_color, label='Train Loss')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(filename)
        plt.close()
    def model_improved(self):
        # if not enough data to compare return True
        if len(self.validation_loss) < 2 or len(self.validation_loss) < 2:
            returnval = True
        else:
            returnval = self.validation_loss[-1] > self.validation_loss[-2]
        if returnval:
            self.min_val_loss = self.validation_loss[-1]
            self.patience_counter = 0
        return returnval
    def is_overfitting(self):
        # if not enough data to compare return False
        if self.model_improved():
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                return True
            else:
                return False

