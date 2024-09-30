import os
import numpy as np
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
            torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.2),
            transforms.RandomRotation((-7, 7)),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalization used for imagenet
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),  # Convert to tensor
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalization used for imagenet
        ])


# def cutout_transform(image, n_holes=1, length=16):
#     h, w = image.size(1), image.size(2)
#     mask = np.ones((h, w), np.float32)
#
#     for _ in range(n_holes):
#         y = np.random.randint(h)
#         x = np.random.randint(w)
#
#         y1 = np.clip(y - length // 2, 0, h)
#         y2 = np.clip(y + length // 2, 0, h)
#         x1 = np.clip(x - length // 2, 0, w)
#         x2 = np.clip(x + length // 2, 0, w)
#
#         mask[y1: y2, x1: x2] = 0.
#
#     mask = torch.from_numpy(mask).expand_as(image)
#     image = image * mask
#     return image


# Download and prepare CIFAR-10 dataset with given transformation
def get_data(batch_size=None):
    traindata = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=get_transform(augment=True))
    trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    testdata = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=get_transform(augment=False))
    testloader = DataLoader(testdata, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return traindata, testdata, trainloader, testloader


def get_labels():
    return 'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
