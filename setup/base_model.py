import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as functional


class BaseCNN(nn.Module):
    def __init__(self, init_type='default'):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.5)
        self.fcl1 = nn.Linear(256 * 4 * 4, 512)
        self.fcl2 = nn.Linear(512, 256)
        self.fcl3 = nn.Linear(256, 128)
        self.fcl4 = nn.Linear(128, 10)
        self._initialize_weights(init_type)

    def forward(self, x):
        x = self.pool(functional.relu(self.conv1(x)))
        x = self.pool(functional.relu(self.conv2(x)))
        x = self.pool(functional.relu(self.conv3(x)))
        # print(x.size())
        x = torch.flatten(x, 1)
        x = functional.relu(self.fcl1(x))
        x = self.dropout(x)
        x = functional.relu(self.fcl2(x))
        x = self.dropout(x)
        x = functional.relu(self.fcl3(x))
        x = self.fcl4(x)
        return x

    def _initialize_weights(self, init_type):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if init_type == 'xavier':
                    init.xavier_uniform_(m.weight)
                elif init_type == 'he':
                    init.kaiming_uniform_(m.weight, nonlinearity='relu')
                elif init_type == 'normal':
                    init.normal_(m.weight)
                elif init_type == 'uniform':
                    init.uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)


model = BaseCNN()
