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
            RandomCrop(32, padding=4),
            # transforms.RandomCrop(32, padding=4),
            # transforms.ColorJitter(brightness=1.0, contrast=0.5, saturation=1, hue=0.1),
            # transforms.RandomRotation(10),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        return transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


# Download and prepare CIFAR-10 dataset with given transformation
def get_data(transform, batch_size=None):
    traindata = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)
    trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True, num_workers=0)

    testdata = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
    testloader = DataLoader(testdata, batch_size=batch_size, shuffle=False, num_workers=0)

    return traindata, testdata, trainloader, testloader


def get_labels():
    return 'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
