import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms
from torchvision.transforms.v2 import RandomCrop


# Convert to PyTorch tensor and normalize between [-1, 1], optional data augmentation
def get_transform(augment=False):
    if augment:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(224),  # Convert to tensor
            # transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),  # Convert to tensor
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


# Download and prepare CIFAR-10 dataset with given transformation
def get_data(batch_size=None):
    traindata = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=get_transform(augment=True))
    trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    testdata = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=get_transform(augment=False))
    testloader = DataLoader(testdata, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    return traindata, testdata, trainloader, testloader


def get_labels():
    return 'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
