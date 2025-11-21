import torch


def calculate_accuracy(model, dataloader, device):
    correct = 0
    total = 0
    model.to(device)
    model.eval()

    with torch.no_grad():
        for data in dataloader:
            images, labels = data["data"], data["labels"]
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def calculate_loss(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data["data"], data["labels"]
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    val_loss /= len(dataloader)
    return val_loss