import copy
import time
import torch
import wandb
import torchmetrics
from setup.config import get_metrics_config, get_device

# Initialize metrics
accuracy_metric, precision_metric, recall_metric, f1_metric = get_metrics_config(get_device())


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, device='cpu'):
    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'train':
                train_loss = epoch_loss
                train_corrects = epoch_acc
                wandb.log({'epoch': epoch + 1, 'training_loss': train_loss, 'training_accuracy': train_corrects})
            else:
                val_loss = epoch_loss
                val_corrects = epoch_acc
                wandb.log({'epoch': epoch + 1, 'validation_loss': val_loss, 'validation_accuracy': val_corrects})

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def train_epoch(epoch, model, trainloader, optimizer, criterion, device='cpu'):
    model.train()
    running_loss = 0.0
    accuracy_metric.reset()
    precision_metric.reset()
    recall_metric.reset()
    f1_metric.reset()

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        outputs, loss = process_batch(optimizer, device, model, criterion, images, labels)
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

    return epoch_loss, epoch_accuracy, epoch_precision, epoch_recall, epoch_f1


def process_batch(optimizer, device, model, criterion, images, labels):
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    return outputs, loss


def update_metrics(outputs, labels):
    accuracy_metric.update(outputs, labels)
    precision_metric.update(outputs, labels)
    recall_metric.update(outputs, labels)
    f1_metric.update(outputs, labels)
