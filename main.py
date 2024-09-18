import sys
from io import StringIO

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision.models import resnet50, ResNet50_Weights, ResNet18_Weights, resnet18
from torchsummary import summary

import wandb
from setup.config import get_wandb_config, get_device
from setup.base_model import BaseCNN
import torch.nn as nn
from setup.data import get_data, get_transform, get_labels
from test import validate_model
from train import train_epoch

# Settings for model and fine-tuning
single_batch = False
conv_layers = 3
linear_layers = 4
pooling = True
batch_norm = True
dropout = 0.5
learning_rate = 0.001
momentum = 0.9
betas = (0.9, 0.99)
epsilon = 1e-8
batch_size = 32
epochs = 1
L2 = 0.0005
augment = True
weights_init = 'xavier'
model_complexity = 'large'
optimizer = 'adam'
model = 'resnet-50'
version = '9.1.1'

# Set up W&B
wandb_config = get_wandb_config(model, model_complexity, learning_rate, betas, epsilon, conv_layers, linear_layers,
                                pooling, batch_norm, dropout, L2, weights_init, augment, optimizer)

wandb.init(project="CIFAR-10-Classification", config=wandb_config, name=version + ' freeze')

# Set up dataset
transform = get_transform(augment)
traindata, testdata, trainloader, testloader = get_data(transform, batch_size)
classes = get_labels()

# Set up model
if model == 'base':
    model = BaseCNN(weights_init)
elif model == 'resnet-18':
    weights = ResNet18_Weights.IMAGENET1K_V1  # or ResNet18_Weights.DEFAULT for updated weights
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
elif model == 'resnet-50':
    weights = ResNet50_Weights.IMAGENET1K_V1  # ResNet50_Weights.DEFAULT for updated weights
    model = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(classes))

# Freeze the fcl layers
for name, param in model.named_parameters():
    if 'fc' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# Calculate which parameters to upgrade which are not frozen
params_to_update = [p for p in model.parameters() if p.requires_grad]

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer initialization
if optimizer == 'adam':
    optimizer = optim.Adam(params_to_update, lr=learning_rate, weight_decay=L2, betas=betas, eps=epsilon)
else:
    optimizer = optim.SGD(params_to_update, lr=learning_rate, momentum=momentum, weight_decay=L2)

# Scheduler for updating learning rates
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
# Run training function (optionally with a single batch)
device = get_device()
epoch_count = 0
model.to(device)

# Set single batch for testing
if single_batch:
    trainloader = [next(iter(trainloader))]
    testloader = trainloader

for epoch in range(epochs):
    train_epoch(epoch_count, model, trainloader, optimizer, criterion, device, scheduler)
    validate_model(epoch, model, testloader, criterion, device)
    epoch_count += 1

# Print summary of the model
model.to(torch.device('cpu'))
print(summary(model, input_size=(3, 32, 32)))


