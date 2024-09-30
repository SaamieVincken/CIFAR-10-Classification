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
from train import train_epoch, set_parameter_requires_grad, train_model

# Settings for model and fine-tuning
single_batch = False
conv_layers = None
linear_layers = None
pooling = None
batch_norm = None
dropout = 0.5
learning_rate = 0.001
momentum = 0.9
betas = (0.9, 0.99)
epsilon = 1e-8
batch_size = 16
epochs = 2
L2 = 0.005
augment = True
weights_init = None
model_complexity = 'large'
optimizer = 'sgd'
model = 'resnet-18'
version = ''
scheduler = ''
feature_extract = False

if __name__ == '__main__':
    # Set up W&B
    wandb_config = get_wandb_config(model, model_complexity, learning_rate, betas, epsilon, conv_layers, linear_layers,
                                    pooling, batch_norm, dropout, L2, weights_init, augment, optimizer, scheduler, batch_size)

    wandb.init(project="CIFAR-10-Classification", config=wandb_config, name=version + ' resnet18',
               notes='')

    # Set up dataset
    traindata, testdata, trainloader, testloader = get_data(batch_size)
    classes = get_labels()

    dataloaders_dict = {'train': trainloader, 'val': testloader}

    # Set up model
    if model == 'base':
        model = BaseCNN(weights_init)
    elif model == 'resnet-18':
        weights = ResNet18_Weights.IMAGENET1K_V1  # Pretrained on imagenet
        model = resnet18(weights=weights)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(classes))
        input_size = 224
    elif model == 'resnet-50':
        weights = ResNet50_Weights.IMAGENET1K_V1  # Pretrained on imagenet
        model = resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, len(classes))

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)

    params_to_update = model.parameters()
    if feature_extract:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)

    optimizer = optim.SGD(params_to_update, lr=learning_rate, momentum=momentum)
    # Loss function
    criterion = nn.CrossEntropyLoss()

    train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=epochs, device=device)