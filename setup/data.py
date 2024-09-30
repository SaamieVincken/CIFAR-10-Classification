import os
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms
from torchvision.transforms.v2 import RandomCrop

mean, std = [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]
# These values are mostly used by researchers as found to very useful in fast convergence
img_size = 224
crop_size = 224


# Convert to PyTorch tensor and normalize between [-1, 1], optional data augmentation
def get_transform(augment=False):
    if augment:
        return transforms.Compose([
            transforms.Resize(img_size),  # , interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            # transforms.CenterCrop(crop_size),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(0.1),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.75, scale=(0.02, 0.1), value=1.0, inplace=False)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
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
