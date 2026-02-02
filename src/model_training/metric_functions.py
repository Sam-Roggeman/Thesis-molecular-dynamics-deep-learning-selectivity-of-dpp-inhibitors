import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix


def calculate_accuracy_and_loss(model, dataloader, criterion, device):
    correct = 0
    total = 0
    current_loss = 0.0

    model.to(device)
    model.eval()

    with torch.no_grad():
        for data in dataloader:
            images, labels = data["data"], data["labels"]
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            current_loss += loss.item()

    accuracy = correct / total
    current_loss /= len(dataloader)

    return accuracy, current_loss

def calculate_precision_recall_f1(model, dataloader, device):
    all_labels = []
    all_predictions = []
    model.to(device)
    model.eval()

    with torch.no_grad():
        for data in dataloader:
            images, labels = data["data"], data["labels"]
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    return precision, recall, f1, conf_matrix

def all_statistics(model, dataloader, criterion, device):
    accuracy, loss = calculate_accuracy_and_loss(model, dataloader, device, criterion)
    precision, recall, f1, conf_matrix = calculate_precision_recall_f1(model=model, dataloader=dataloader, device=device)

    return {
        "accuracy": accuracy,
        "loss": loss,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": conf_matrix
    }