import os

from src.model_training.metric_functions import all_statistics
import matplotlib.pyplot as plt
from src.model_training.LabelEncoder import LabelEncoder
from sklearn.metrics import ConfusionMatrixDisplay


def model_testing(model, testloader, criterion, device, output_dir, max_batches=None):
    statistics = all_statistics(
        model=model,
        dataloader=testloader,
        device=device,
        criterion=criterion,
        max_batches=max_batches,
    )
    labelencoder = LabelEncoder()
    label_classes = labelencoder.get_classes()
    print(statistics)


    cm = statistics.pop("confusion_matrix")
    plot_cm(cm, label_classes)
    plt.savefig(os.path.join(output_dir,"confusion_matrix.png"))

def plot_cm(cm, label_classes):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    # rotate x and y axis labels
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
