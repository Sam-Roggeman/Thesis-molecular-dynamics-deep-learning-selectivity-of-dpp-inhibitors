import os
import torch

from src.model_training.metric_functions import all_statistics, calculate_statistics
import matplotlib.pyplot as plt
from src.model_training.LabelEncoder import LabelEncoder
from sklearn.metrics import ConfusionMatrixDisplay


def model_testing(
    model,
    testloader,
    criterion,
    device,
    output_dir,
    max_batches=None,
    split_name="test",
    batch_preparation_fn=None,
    use_mixed_precision=False,
    amp_dtype=torch.bfloat16,
):
    """Test a model on a given test loader and compute various statistics.
    Args:
        model: The PyTorch model to be tested.
        testloader: DataLoader for the test dataset.
        criterion: Loss function to compute the loss.
        device: Device to run the model on (e.g., 'cuda' or 'cpu').
        output_dir: Directory to save the results and statistics.
        max_batches: Maximum number of batches to process. If None, process all batches.
        split_name: Name of the data split (e.g., 'test', 'validation').
        batch_preparation_fn: Optional function to prepare batches before feeding them to the model.
        use_mixed_precision: Whether to use mixed precision for inference.
        amp_dtype: The data type for automatic mixed precision (e.g., torch.float16, torch.bfloat16).
    """
    statistics = calculate_statistics(
        model=model,
        dataloader=testloader,
        device=device,
        criterion=criterion,
        max_batches=max_batches,
        batch_preparation_fn=batch_preparation_fn,
        use_mixed_precision=use_mixed_precision,
        amp_dtype=amp_dtype,
    )
    labelencoder = LabelEncoder()
    label_classes = labelencoder.get_classes()
    print(statistics)

    # save statistics to file
    with open(os.path.join(output_dir, "statistics.txt"), "w") as f:
        print("Statistics:", file=f)
        print(statistics, file=f)
            


    cm = statistics.pop("confusion_matrix")
    cm_normalized = statistics.pop("confusion_matrix_normalized")
    title = f"Confusion Matrix for {model.__class__.__name__} on {split_name} split"
    plot_cm(cm, cm_normalized, label_classes, title)
    plt.savefig(os.path.join(output_dir,"confusion_matrix.png"))

def plot_cm(cm, cm_normalized, label_classes, title):
    # display both normalized and unnormalized confusion matrix in the same plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_classes)
    disp.plot(cmap=plt.cm.Blues, ax=axes[0])
    axes[0].set_title("Confusion Matrix (Counts)")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=label_classes)
    disp.plot(cmap=plt.cm.Blues, ax=axes[1])
    axes[1].set_title("Confusion Matrix (Normalized)")
    title_top_margin = 0.02
    fig.suptitle(title, fontsize=12, y=1.0 - title_top_margin)
    # rotate all x and y labels of the subplots by 45 degrees for better readability
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=45)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
        ax.tick_params(axis="y", labelrotation=45)

    # Reserve extra margins so rotated labels and title remain inside the figure.
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 1.0-title_top_margin ))
    fig.subplots_adjust(bottom=0.2)
    

def _test_cm():
    # 5x5 confusion matrix example ndarray of shape (5, 5) with values 0 for testing
    import numpy as np
    matrix =  [[0 for _ in range(5)] for _ in range(5)]
    cm = np.array(matrix)
    cm_normalized = np.array(matrix)
    label_classes = ["Class 0", "Class 1", "Class 2", "Class 3", "Class 4"]
    plot_cm(cm, cm_normalized, label_classes, "Test Confusion Matrix")
    # save the plot to a file
    plt.savefig("test_confusion_matrix.png")



if __name__ == "__main__":
    _test_cm()