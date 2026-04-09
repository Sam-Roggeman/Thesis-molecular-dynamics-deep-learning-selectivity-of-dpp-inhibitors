import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

def calculate_statistics(model, dataloader, criterion, device, max_batches: int | None = None, statistics_to_compute: list[str] | None = None) -> dict[str, float]:
    """Calculate specified statistics for a model on a given dataloader. Loop over the dataloader and compute the specified statistics for each batch, then average them over the entire dataloader.
    Args:
        model (_type_): _model to evaluate
        dataloader (_type_): dataloader to evaluate on
        criterion (_type_): loss function to use for loss calculation
        device (_type_): device to use for evaluation
        max_batches (int | None, optional): maximum number of batches to evaluate. Defaults to None.
        statistics_to_compute (list[str] | None, optional): list of statistics to compute. Defaults to None.

    Returns:
        dict[str, float]: dictionary containing the computed statistics
    """
    if statistics_to_compute is None:
        statistics_to_compute = ["accuracy", "loss", "precision", "recall", "f1_score", "confusion_matrix"]
    
    all_labels = []
    all_predictions = []
    total_loss = 0.0
    num_batches = 0

    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            print(f"Processing batch {batch_idx + 1}...", end="\r")
            if max_batches is not None and batch_idx >= max_batches:
                break
            images, labels = data["data"], data["labels"]
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            num_batches += 1

            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    statistics = {}
    if "accuracy" in statistics_to_compute:
        statistics["accuracy"] = accuracy_score(all_labels, all_predictions)
    if "loss" in statistics_to_compute:
        statistics["loss"] = total_loss / max(num_batches, 1)
    if "precision" in statistics_to_compute:
        statistics["precision"] = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    if "recall" in statistics_to_compute:
        statistics["recall"] = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    if "f1_score" in statistics_to_compute:
        statistics["f1_score"] = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    if "confusion_matrix" in statistics_to_compute:
        statistics["confusion_matrix"] = confusion_matrix(all_labels, all_predictions, labels=list(range(5)))
        statistics["confusion_matrix_normalized"] = confusion_matrix(all_labels, all_predictions, normalize='true', labels=list(range(5)))

    return statistics

def calculate_accuracy_and_loss(model, dataloader, criterion, device, max_batches: int | None = None):
    correct = 0
    total = 0
    current_loss = 0.0
    num_batches = 0

    model.to(device)
    model.eval()

    with torch.no_grad():
        # iterate over streaming dataloader
        for batch_idx, data in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images, labels = data["data"], data["labels"]
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            current_loss += loss.item()
            num_batches += 1

    accuracy = correct / max(total, 1)
    current_loss /= max(num_batches, 1)

    return accuracy, current_loss

def calculate_precision_recall_f1(model, dataloader, device, max_batches: int | None = None):
    all_labels = []
    all_predictions = []
    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images, labels = data["data"], data["labels"]
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_predictions,normalize='true', labels=list(range(5)))

    return precision, recall, f1, conf_matrix

def all_statistics(model, dataloader, criterion, device, max_batches: int | None = None) -> dict[str, float]:
    accuracy, loss = calculate_accuracy_and_loss(
        model=model,
        dataloader=dataloader,
        device=device,
        criterion=criterion,
        max_batches=max_batches,
    )
    precision, recall, f1, conf_matrix = calculate_precision_recall_f1(
        model=model,
        dataloader=dataloader,
        device=device,
        max_batches=max_batches,
    )

    return {
        "accuracy": accuracy,
        "loss": loss,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": conf_matrix
    }