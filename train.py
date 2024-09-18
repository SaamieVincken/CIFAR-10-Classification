import torch
import wandb
import torchmetrics
from setup.config import get_metrics_config, get_device

# Initialize metrics
accuracy_metric, precision_metric, recall_metric, f1_metric = get_metrics_config(get_device())


def train_epoch(epoch, model, trainloader, optimizer, criterion, device='cpu', scheduler=None, L1=None):
    model.train()
    running_loss = 0.0
    accuracy_metric.reset()
    precision_metric.reset()
    recall_metric.reset()
    f1_metric.reset()

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        outputs, loss = process_batch(optimizer, device, model, criterion, images, labels, L1)
        update_metrics(outputs, labels)
        running_loss += loss.item()

    # Calculate metrics per epoch
    epoch_loss = running_loss / len(trainloader)
    epoch_accuracy = accuracy_metric.compute().item()
    epoch_precision = precision_metric.compute().item()
    epoch_recall = recall_metric.compute().item()
    epoch_f1 = f1_metric.compute().item()

    # Log metrics to W&B
    wandb.log({
        'epoch': epoch + 1,
        'training_loss': epoch_loss,
        'training_accuracy': epoch_accuracy,
        'training_precision': epoch_precision,
        'training_recall': epoch_recall,
        'training_f1_score': epoch_f1,
    })

    scheduler.step()

    return epoch_loss, epoch_accuracy, epoch_precision, epoch_recall, epoch_f1


def process_batch(optimizer, device, model, criterion, images, labels, L1=None):
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)

    if L1:
        # Add L1 regularization
        L1_loss = sum(p.abs().sum() for p in model.parameters())
        loss += L1 * L1_loss

    loss.backward()
    optimizer.step()
    return outputs, loss


def update_metrics(outputs, labels):
    accuracy_metric.update(outputs, labels)
    precision_metric.update(outputs, labels)
    recall_metric.update(outputs, labels)
    f1_metric.update(outputs, labels)
