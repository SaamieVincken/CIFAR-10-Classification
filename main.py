import sys
import time
from io import StringIO
import tensorflow as tf
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision.models import resnet50, ResNet50_Weights, ResNet18_Weights, resnet18
from torchsummary import summary

import wandb
from setup.config import get_wandb_config, get_device
from setup.base_model import BaseCNN
import torch.nn as nn
from setup.data import get_data, get_labels
from test import validate_model
from train import train_epoch

# Settings for model and fine-tuning
single_batch = False
conv_layers = None
linear_layers = None
pooling = None
batch_norm = None
dropout = 0.5
learning_rate = 0.01
momentum = 0.9
betas = (0.9, 0.99)
epsilon = 1e-8
batch_size = 32
epochs = 10
L2 = 0.0005
augment = True
weights_init = 'xavier'
model_complexity = 'large'
optimizer = 'sgd'
model = 'resnet-18'
version = '9.2.6.1'

if __name__ == '__main__':
    # Set up W&B
    wandb_config = get_wandb_config(model, model_complexity, learning_rate, betas, epsilon, conv_layers, linear_layers,
                                    pooling, batch_norm, dropout, L2, weights_init, augment, optimizer)

    wandb.init(project="CIFAR-10-Classification", config=wandb_config, name=version + ' resnet18', notes='Same as 9.2.6 but 20 epochs')

    # Set up dataset
    traindata, testdata, trainloader, testloader = get_data(batch_size)
    classes = get_labels()

    # Set up model
    if model == 'base':
        model = BaseCNN(weights_init)
    elif model == 'resnet-18':
        weights = ResNet18_Weights.IMAGENET1K_V1  # Pretrained on imagenet
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, len(classes))
    elif model == 'resnet-50':
        weights = ResNet50_Weights.IMAGENET1K_V1  # Pretrained on imagenet
        model = resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, len(classes))

    # Freeze layers selectively
    for name, param in model.named_parameters():
        if 'fc' not in name and 'layer3' not in name and 'layer4' not in name:
            param.requires_grad = False

    # Calculate which parameters to update which are not frozen
    params_to_update = [p for p in model.parameters() if p.requires_grad]

    for name, param in model.named_parameters():
        print(f"{name}: {'requires_grad' if param.requires_grad else 'frozen'}")

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer initialization
    if optimizer == 'adam':
        optimizer = optim.Adam(params_to_update, lr=learning_rate, weight_decay=L2, betas=betas, eps=epsilon)
    else:
        optimizer = optim.SGD(params_to_update, lr=learning_rate, momentum=momentum, weight_decay=L2)

    # Print summary of the model
    model.to(torch.device('cpu'))
    print(summary(model, input_size=(3, 32, 32)))

    device = get_device()
    epoch_count = 0
    model.to(device)

    # Set single batch for testing
    if single_batch:
        trainloader = [next(iter(trainloader))]
        testloader = trainloader

    # Training/validation function
    for epoch in range(epochs):
        train_epoch(epoch_count, model, trainloader, optimizer, criterion, device)
        val_epoch_loss, val_epoch_accuracy, val_epoch_precision, val_epoch_recall, val_epoch_f1 =\
            validate_model(epoch, model, testloader, criterion, device)
        epoch_count += 1


    # scheduler.step()
    # Scheduler for updating learning rates (StepLR or ReduceLROnPlateau)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)