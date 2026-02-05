import os

import torch
import matplotlib.pyplot as plt
from typing import List

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
            "train_loss": self.train_loss,
            "validation_loss": self.validation_loss
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
        self.validation_loss = metrics_dict.get("validation_loss", [])




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
        if len(self.validation_loss) < 2:
            return True
        # val loss decreased
        elif self.validation_loss[-1] < self.min_val_loss:
            self.min_val_loss = self.validation_loss[-1]
            self.patience_counter = 0
            return True
        return False

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
    @staticmethod
    def compare_metrics(metrics: List['Metrics'], names: List[str] = None, path="./output/comparison_plots/last_comparison.png"):
        ms: Metrics
        max_epochs = max(len(ms.training_accuracy) for ms in metrics)
        epochs = range(1, max_epochs + 1)
        min_loss = min(min(ms.train_loss + ms.validation_loss) for ms in metrics)
        max_loss = max(max(ms.train_loss + ms.validation_loss) for ms in metrics)
        loss_diff = max_loss - min_loss
        # add some margin to min and max loss
        min_loss -= 0.1 * loss_diff
        max_loss += 0.1 * loss_diff

        # assign each model stat a different color
        for ms in metrics:
            color = plt.colormaps.get_cmap('tab10')(metrics.index(ms) % 10)
            ms.color = color
    
        # create subplots
        fig, axs = plt.subplots(2, 2)
        # set margin between subplots
        fig.subplots_adjust(hspace=0.1, wspace=0.05)

        # Training accuracy plot
        active_ax = axs[0, 0]
        for ms in metrics:
            y = ms.training_accuracy + [None] * (max_epochs - len(ms.training_accuracy))
            active_ax.plot(epochs, y, color=ms.color, label=names[metrics.index(ms)])
        active_ax.set_title('')
        active_ax.set(xlabel='Epochs', ylabel='Accuracy')
        active_ax.set_ylim(0, 100)
        active_ax.set_title('Training')

        active_ax = axs[0, 1]
        # Validation accuracy plot
        for ms in metrics:
            y = ms.validation_accuracy + [None] * (max_epochs - len(ms.validation_accuracy))
            active_ax.plot(epochs, y, color=ms.color, label=names[metrics.index(ms)])
        active_ax.set_title('')
        active_ax.set(xlabel='Epochs', ylabel='Accuracy')
        active_ax.set_ylim(0, 100)
        active_ax.set_title('Validation')

        active_ax = axs[1, 0]
        # Training loss plot
        for ms in metrics:
            y = ms.train_loss + [None] * (max_epochs - len(ms.train_loss))
            active_ax.plot(epochs, y, color=ms.color, label=names[metrics.index(ms)])
        active_ax.set_ylim(min_loss, max_loss)
        active_ax.set(xlabel='Epochs', ylabel='Loss')

        active_ax = axs[1, 1]
        # validation loss plot
        for ms in metrics:
            y = ms.validation_loss + [None] * (max_epochs - len(ms.validation_loss))
            active_ax.plot(epochs, y, color=ms.color, label=names[metrics.index(ms)])
        active_ax.set_ylim(min_loss, max_loss)
        active_ax.set(xlabel='Epochs', ylabel='Loss')

        # xax ticks every 5 epochs or at least 3 ticks
        for x in [0,1]:
            axs[1, x].set_xticks(range(1, max_epochs + 1, max(1, max_epochs // 10)))

        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axs.flat:
            ax.label_outer()
        # Add legend to thefigure
        handles, labels = active_ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=max(len(metrics),4))

        dirpath = os.path.dirname(path)
        # create path if it does not exist
        os.makedirs(dirpath, exist_ok=True)
        # print
        print(f"Saving comparison plot to {path}")
        fig.savefig(path)
if __name__ == "__main__":
    # Example usage
    metrics = Metrics(patience=5)
    metrics.update(0.8, 0.5, 0.75, 0.6)
    metrics.update(0.85, 0.4, 0.78, 0.55)
    metrics.update(0.9, 0.3, 0.8, 0.5)

    metrics2 = Metrics(patience=5)
    metrics2.update(0.7, 0.6, 0.65,
                    0.7)
    metrics2.update(0.75, 0.5, 0.68, 0.65)
    metrics2.update(0.8, 0.4, 0.7, 0.6)
    metrics2.update(0.8, 0.4, 0.7, 0.6)
    Metrics.compare_metrics([metrics, metrics2], names=["Model 1", "Model 2"])
    plt.show()
