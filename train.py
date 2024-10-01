import copy
import time
import torch
import wandb
import torchmetrics
from setup.config import get_metrics_config, get_device

# Initialize metrics
accuracy_metric, precision_metric, recall_metric, f1_metric = get_metrics_config(get_device())


def train_model(model, dataloaders, criterion, optimizer, classes, num_epochs=5, device='cpu'):
    best_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            running_loss = 0.0
            accuracy_metric.reset()
            precision_metric.reset()
            recall_metric.reset()
            f1_metric.reset()
            true_labels = []
            predicted_labels = []
            correct = torch.zeros(len(classes), dtype=torch.int64)
            total = torch.zeros(len(classes), dtype=torch.int64)

            if phase == 'train':
                model.train()
            else:
                model.eval()

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, predictions = torch.max(outputs, 1)

                    # backward + optimize for training
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    update_metrics(predictions, labels)

                    true_labels.extend(labels.cpu().numpy())
                    predicted_labels.extend(predictions.cpu().numpy())

                    # Count correct predictions and total instances per class
                    for i in range(len(classes)):
                        class_mask = (labels == i)
                        correct[i] += (predictions[class_mask] == i).sum().item()
                        total[i] += class_mask.sum().item()

            # Calculate per-class accuracy
            per_class_accuracy = correct.float() / total.float()
            per_class_accuracy[total == 0] = 0

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_accuracy = accuracy_metric.compute().item()
            epoch_precision = precision_metric.compute().item()
            epoch_recall = recall_metric.compute().item()
            epoch_f1 = f1_metric.compute().item()

            if phase == 'train':
                wandb.log({
                    'epoch': epoch + 1,
                    'training_loss': epoch_loss,
                    'training_accuracy': epoch_accuracy,
                    'training_precision': epoch_precision,
                    'training_recall': epoch_recall,
                    'training_f1_score': epoch_f1,
                })
            else:
                wandb.log({
                    'epoch': epoch + 1,
                    'validation_loss': epoch_loss,
                    'validation_accuracy': epoch_accuracy,
                    'validation_precision': epoch_precision,
                    'validation_recall': epoch_recall,
                    'validation_f1_score': epoch_f1,
                    "predictions": wandb.plot.confusion_matrix(
                        preds=predicted_labels,
                        y_true=true_labels,
                        class_names=classes
                    )
                })

            # Log per-class accuracy only for validation
            for i, class_name in enumerate(classes):
                wandb.log({f'Accuracy/{class_name}': per_class_accuracy[i].item()})

            # deep copy the model
            if phase == 'val' and epoch_accuracy > best_acc:
                best_acc = epoch_accuracy
                best_weights = copy.deepcopy(model.state_dict())

    # load best model weights
    model.load_state_dict(best_weights)


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def update_metrics(outputs, labels):
    accuracy_metric.update(outputs, labels)
    precision_metric.update(outputs, labels)
    recall_metric.update(outputs, labels)
    f1_metric.update(outputs, labels)

