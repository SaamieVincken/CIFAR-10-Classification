import sys
import time
from io import StringIO
import tensorflow as tf
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, OneCycleLR
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
learning_rate = 0.0001
momentum = 0.9
betas = (0.9, 0.99)
epsilon = 1e-8
batch_size = 128
epochs = 20
L2 = 0.005
augment = True
weights_init = None
model_complexity = 'large'
optimizer = 'adam'
model = 'resnet-18'
version = ''
scheduler = 'ReduceLROnPlateau'

if __name__ == '__main__':
    # Set up W&B
    wandb_config = get_wandb_config(model, model_complexity, learning_rate, betas, epsilon, conv_layers, linear_layers,
                                    pooling, batch_norm, dropout, L2, weights_init, augment, optimizer, scheduler, batch_size)

    wandb.init(project="CIFAR-10-Classification", config=wandb_config, name=version + ' resnet18',
               notes='')

    # Set up dataset
    traindata, testdata, trainloader, testloader = get_data(batch_size)
    classes = get_labels()

    # Set up model
    if model == 'base':
        model = BaseCNN(weights_init)
    elif model == 'resnet-18':
        weights = ResNet18_Weights.IMAGENET1K_V1  # Pretrained on imagenet
        model = resnet18(weights=weights)
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(model.fc.in_features, len(classes))
        )
        model.maxpool = nn.Identity()  # Remove the max pooling layer
    elif model == 'resnet-50':
        weights = ResNet50_Weights.IMAGENET1K_V1  # Pretrained on imagenet
        model = resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, len(classes))

    # Freeze layers
    for name, param in model.named_parameters():
        if 'fc' not in name and 'layer3' not in name and 'layer4' not in name:
            param.requires_grad = False

    # Calculate which parameters to update which are not frozen
    params_to_update = [p for p in model.parameters() if p.requires_grad]

    # Loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Optimizer initialization
    if optimizer == 'adam':
        optimizer = optim.Adam(params_to_update, lr=learning_rate, weight_decay=L2, betas=betas, eps=epsilon)
    elif optimizer == 'adamW':
        optimizer = optim.AdamW(params_to_update, lr=learning_rate, weight_decay=L2)
    else:
        optimizer = optim.SGD(params_to_update, lr=learning_rate, momentum=momentum, weight_decay=L2)

    # Scheduler initialization
    if scheduler == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.1)
    elif scheduler == 'OneCycleLR':
        scheduler = OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=len(trainloader), epochs=epochs)

    # Print summary of the model
    # model.to(torch.device('cpu'))
    # print(summary(model, input_size=(3, 32, 32)))

    # for name, param in model.named_parameters():
    # print(f"{name}: {'requires_grad' if param.requires_grad else 'frozen'}")

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
        val_epoch_loss, val_epoch_accuracy, val_epoch_precision, val_epoch_recall, val_epoch_f1 = \
            validate_model(epoch, model, testloader, criterion, classes, device)

        # ToDo remove epoch loss depending on scheduler
        scheduler.step(val_epoch_loss)
        epoch_count += 1
