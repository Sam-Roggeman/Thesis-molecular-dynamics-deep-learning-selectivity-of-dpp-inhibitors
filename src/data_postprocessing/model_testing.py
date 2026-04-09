import os

from src.model_training.metric_functions import all_statistics, calculate_statistics
import matplotlib.pyplot as plt
from src.model_training.LabelEncoder import LabelEncoder
from sklearn.metrics import ConfusionMatrixDisplay


def model_testing(model, testloader, criterion, device, output_dir, max_batches=None):
    statistics = calculate_statistics(
        model=model,
        dataloader=testloader,
        device=device,
        criterion=criterion,
        max_batches=max_batches,
    )
    labelencoder = LabelEncoder()
    label_classes = labelencoder.get_classes()
    print(statistics)

    # save statistics to file
    with open(os.path.join(output_dir, "statistics.txt"), "w") as f:
        print("Statistics:", file=f)
            


    cm = statistics.pop("confusion_matrix")
    cm_normalized = statistics.pop("confusion_matrix_normalized")
    plot_cm(cm, label_classes)
    plt.savefig(os.path.join(output_dir,"confusion_matrix.png"))
    plot_cm(cm_normalized, label_classes)
    plt.savefig(os.path.join(output_dir,"confusion_matrix_normalized.png"))

def plot_cm(cm, label_classes):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    # rotate x and y axis labels
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    # zoom out to fit the labels
    plt.tight_layout()
