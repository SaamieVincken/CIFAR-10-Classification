import copy
import time
import torch
import wandb
import torchmetrics
from setup.config import get_metrics_config, get_device

# Initialize metrics
accuracy_metric, precision_metric, recall_metric, f1_metric = get_metrics_config(get_device())


def train_model(model, dataloaders, criterion, optimizer, classes, num_epochs=5, device='cpu', patience=3):
    best_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epochs_no_improve = 0
    min_delta = 0.001

    for epoch in range(num_epochs):
        for phase in ['training', 'validation']:
            running_loss = 0.0
            accuracy_metric.reset()
            precision_metric.reset()
            recall_metric.reset()
            f1_metric.reset()
            true_labels = []
            predicted_labels = []
            correct = torch.zeros(len(classes), dtype=torch.int64)
            total = torch.zeros(len(classes), dtype=torch.int64)

            if phase == 'training':
                model.train()
            else:
                model.eval()

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'training'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, predictions = torch.max(outputs, 1)

                    # Backward + optimize only if in training phase
                    if phase == 'training':
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

            # Logging metrics
            metrics = {
                'epoch': epoch + 1,
                f'{phase}_loss': epoch_loss,
                f'{phase}_accuracy': epoch_accuracy,
                f'{phase}_precision': epoch_precision,
                f'{phase}_recall': epoch_recall,
                f'{phase}_f1_score': epoch_f1,
            }

            # Per class accuracy
            for i, class_name in enumerate(classes):
                metrics[f'{phase}_accuracy/{class_name}'] = per_class_accuracy[i].item()

            # Confusion matrix only for the validation phase
            if phase == 'validation':
                metrics["predictions"] = wandb.plot.confusion_matrix(
                    preds=predicted_labels,
                    y_true=true_labels,
                    class_names=classes
                )

            wandb.log(metrics)

            # Early stopping check
            if phase == 'validation' and epoch_accuracy > best_acc + min_delta:
                best_acc = epoch_accuracy
                best_weights = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0  # Reset the counter when improving
            elif phase == 'validation':
                epochs_no_improve += 1

            # Early stopping
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {patience} epochs with no improvement.")
                model.load_state_dict(best_weights)
                return model

    # Load best model weights
    model.load_state_dict(best_weights)
    return model


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def update_metrics(outputs, labels):
    accuracy_metric.update(outputs, labels)
    precision_metric.update(outputs, labels)
    recall_metric.update(outputs, labels)
    f1_metric.update(outputs, labels)
