import torch
import torch.optim as optim
import wandb
from setup.config import get_wandb_config
from setup.base_model import BaseCNN
import torch.nn as nn
from setup.data import get_data, get_transform, get_labels
from train import train_model

# Set up W&B
wandb_config = get_wandb_config()
wandb.init(project="CIFAR-10-Classification", config=wandb_config, name="test v0.1")

# Set up dataset
transform = get_transform()
traindata, testdata, trainloader, testloader = get_data(transform, 32)
classes = get_labels()

# Set up base model
model = BaseCNN()

# Loss function
criterion = nn.CrossEntropyLoss()

# SGD optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


# ToDo: Try out new mps accelerated pytorch training for mac
# # Define device
# if torch.backends.mps.is_available():
#     # For M3 silicon
#     device = 'mps'
# else:
#     device = 'cpu'

# Run training function with a single batch
for epoch in range(1):
    train_model(model, trainloader, optimizer, criterion, device='cpu')

