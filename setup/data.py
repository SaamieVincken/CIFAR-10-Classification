import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms


# Convert to PyTorch tensor and normalize between [-1, 1]
def get_transform():
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
