import torch
import wandb
import torchmetrics
from setup.config import get_metrics_config

# Initialize metrics
accuracy_metric, precision_metric, recall_metric, f1_metric = get_metrics_config()


# Training function
def train_model(model, trainloader, optimizer, criterion, device='cpu'):

    model.train()

    # Temp: This code is only used for the first part of the assignment
    # Get a single batch of data (images and labels)
    data_iter = iter(trainloader)
    images, labels = next(data_iter)

    # Move data to device
    images, labels = images.to(device), labels.to(device)

    # Reset gradients
    optimizer.zero_grad()

    # Get model output
    outputs = model(images)

    # Get the loss
    loss = criterion(outputs, labels)

    # Calculate the gradient
    loss.backward()

    # Update learning weights
    optimizer.step()

    # Calculate metrics
    accuracy = accuracy_metric(outputs, labels)
    precision = precision_metric(outputs, labels)
    recall = recall_metric(outputs, labels)
    f1_score = f1_metric(outputs, labels)

    # Log metrics to W&B
    wandb.log({
        'loss': loss.item(),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    })
