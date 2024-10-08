import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms

mean, std = [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]
img_size = 128


def get_transform(augment=False):
    if augment:
        return transforms.Compose([
            transforms.Resize(img_size),
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
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])


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
