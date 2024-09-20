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

# Processing info:
# devices = tf.config.list_physical_devices()
# print("\nDevices: ", devices)
#
# if torch.backends.mps.is_available():
#     mps_device = torch.device("mps")
#     x = torch.ones(1, device=mps_device)
#     print(x)
#
#     start_time = time.time()
#
#     # synchronize time with cpu, otherwise only time for dogfooding data to gpu would be measured
#     torch.mps.synchronize()
#
#     a = torch.ones(4000, 4000, device="mps")
#     for _ in range(200):
#         a += a
#
#     elapsed_time = time.time() - start_time
#     print("GPU Time: ", elapsed_time)

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
batch_size = 128
epochs = 5
L2 = 0.0005
augment = True
weights_init = 'xavier'
model_complexity = 'large'
optimizer = 'adam'
model = 'resnet-50'
version = '9.1.3'

if __name__ == '__main__':
    # Set up W&B
    wandb_config = get_wandb_config(model, model_complexity, learning_rate, betas, epsilon, conv_layers, linear_layers,
                                    pooling, batch_norm, dropout, L2, weights_init, augment, optimizer)

    wandb.init(project="CIFAR-10-Classification", config=wandb_config, name=version + ' freeze')

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
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 10),  # Adjust the final layer for CIFAR-10
            nn.Softmax(dim=1)  # Add softmax activation
        )
        # weights = ResNet50_Weights.IMAGENET1K_V1  # Pretrained on imagenet
        # model = resnet50(weights=weights)
        # model.fc = nn.Linear(model.fc.in_features, len(classes))

    # # Freeze the pretrained weights
    # for name, param in model.named_parameters():
    #     if 'fc' not in name:  # except for final fcl
    #         param.requires_grad = False

    # Freeze layers selectively
    for name, param in model.named_parameters():
        if 'fc' not in name and 'layer4' not in name:  # Unfreeze last block
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

    # Scheduler for updating learning rates (StepLR or ReduceLROnPlateau)
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

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
        scheduler.step(val_epoch_loss)
        epoch_count += 1

    # Print summary of the model
    model.to(torch.device('cpu'))
    print(summary(model, input_size=(3, 32, 32)))
