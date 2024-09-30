import torch
import torchmetrics

DEVICE = None


def get_wandb_config(model='base', model_complexity='base model', learning_rate=None,
                     betas=None, epsilon=None, conv_layers=None, linear_layers=None, pooling=False, batch_norm=False,
                     dropout=None, L2=None, weights_init=None, data_augmentation=True, optimizer=None, scheduler=None, batch_size=None):
    # Return all configurations for wandb logging
    return {
        'architecture': 'CNN',
        'dataset': 'CIFAR-10',
        'model': model,
        'model_complexity': model_complexity,
        'learning_rate': learning_rate,
        'betas': betas,
        'epsilon': epsilon,
        'conv_layers': conv_layers,
        'linear_layers': linear_layers,
        'pooling': pooling,
        'batch_norm': batch_norm,
        'dropout': dropout,
        'L2': L2,
        'weights_init': weights_init,
        'data_augmentation': data_augmentation,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'batch_size': batch_size,
    }


def get_metrics_config(device):
    accuracy_metric = torchmetrics.Accuracy(task='multiclass', num_classes=10).to(device)
    precision_metric = torchmetrics.Precision(task='multiclass', num_classes=10).to(device)
    recall_metric = torchmetrics.Recall(task='multiclass', num_classes=10).to(device)
    f1_metric = torchmetrics.F1Score(task='multiclass', num_classes=10).to(device)
    return accuracy_metric, precision_metric, recall_metric, f1_metric


def get_device():
    global DEVICE
    if DEVICE is None:
        if torch.backends.mps.is_available():
            DEVICE = torch.device('mps')  # Apple Silicon (Metal Performance Shaders)
        elif torch.cuda.is_available():
            DEVICE = torch.device('cuda')  # GPU
        else:
            DEVICE = torch.device('cpu')  # CPU
    return DEVICE


