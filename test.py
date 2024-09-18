import torch
import wandb
import torchmetrics
from setup.config import get_metrics_config, get_device

# Initialize metrics
accuracy_metric, precision_metric, recall_metric, f1_metric = get_metrics_config(get_device())


def validate_model(epoch, model, testloader, criterion, device='cpu'):
    model.eval()
    running_loss = 0.0
    accuracy_metric.reset()
    precision_metric.reset()
    recall_metric.reset()
    f1_metric.reset()

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Update metrics
            update_metrics(outputs, labels)

    # Calculate metrics
    epoch_loss = running_loss / len(testloader)
    epoch_accuracy = accuracy_metric.compute().item()
    epoch_precision = precision_metric.compute().item()
    epoch_recall = recall_metric.compute().item()
    epoch_f1 = f1_metric.compute().item()

    # Log metrics to W&B
    wandb.log({
        'epoch': epoch + 1,
        'validation_loss': epoch_loss,
        'validation_accuracy': epoch_accuracy,
        'validation_precision': epoch_precision,
        'validation_recall': epoch_recall,
        'validation_f1_score': epoch_f1,
    })

    return epoch_loss, epoch_accuracy, epoch_precision, epoch_recall, epoch_f1


def update_metrics(outputs, labels):
    accuracy_metric.update(outputs, labels)
    precision_metric.update(outputs, labels)
    recall_metric.update(outputs, labels)
    f1_metric.update(outputs, labels)
