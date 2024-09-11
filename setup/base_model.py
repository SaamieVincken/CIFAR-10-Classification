import torch
import torch.nn as nn
import torch.nn.functional as functional


class BaseCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv. layer 1
        # in: number of input channels (RGB = 3)
        # out: number of filters (32 filters for initial setup)
        # kernel_size = size of filter (3x3 for initial setup)
        # stride = speed the filter moves across an image (stride is 1 = 1 pixel at a time)
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1)
        # Max-pool decr. by factor 2
        self.pool = nn.MaxPool2d(2, 2)
        # Conv. layer 2
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1)
        # Max-pool decr. by factor 2
        self.pool = nn.MaxPool2d(2, 2)
        # Fully connected layers for classification:
        self.fcl1 = nn.Linear(64 * 6 * 6, 200)
        self.fcl2 = nn.Linear(200, 100)
        # Output features: 10 -> Number of categories in CIFAR-10
        self.fcl3 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.pool(functional.relu(self.conv1(x)))
        x = self.pool(functional.relu(self.conv2(x)))
        # Flatten the feature maps
        x = torch.flatten(x, 1)
        x = functional.relu(self.fcl1(x))
        x = functional.relu(self.fcl2(x))
        x = self.fcl3(x)
        return x


model = BaseCNN()

