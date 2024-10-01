import torch.optim as optim
from torchvision.models import  ResNet18_Weights, resnet18
import wandb
from setup.config import get_wandb_config, get_device
from setup.base_model import BaseCNN
import torch.nn as nn
from setup.data import get_data, get_labels
from train import set_parameter_requires_grad, train_model

# Settings for model and fine-tuning
dropout = 0.5
learning_rate = 0.001
momentum = 0.9
batch_size = 16
epochs = 10
augment = True
weights_init = None
model_complexity = 'large'
optimizer = 'sgd'
model = 'resnet-18'
version = '13.6'
scheduler = ''
feature_extract = False

if __name__ == '__main__':
    # Set up W&B
    wandb_config = get_wandb_config(model, model_complexity, learning_rate, betas=None, epsilon=None, conv_layers=None, linear_layers=None,
                                    pooling=None, batch_norm=None, dropout=None, L2=None, weights_init=None, data_augmentation=augment,
                                    optimizer=optimizer, scheduler=None, batch_size=batch_size)

    wandb.init(project="CIFAR-10-Classification", config=wandb_config, name=version, notes='')

    # Set up dataset
    traindata, testdata, trainloader, testloader = get_data(batch_size)
    classes = get_labels()

    dataloaders_dict = {'training': trainloader, 'validation': testloader}

    # Set up model
    if model == 'base':
        model = BaseCNN(weights_init)
    elif model == 'resnet-18':
        weights = ResNet18_Weights.IMAGENET1K_V1  # Pretrained on imagenet
        model = resnet18(weights=weights)
        set_parameter_requires_grad(model, feature_extract)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, len(classes))

    device = get_device()
    model = model.to(device)

    params_to_update = [param for param in model.parameters() if not feature_extract or param.requires_grad]

    optimizer = optim.SGD(params_to_update, lr=learning_rate, momentum=momentum)
    criterion = nn.CrossEntropyLoss()
    train_model(model, dataloaders_dict, criterion, optimizer, classes, num_epochs=epochs, device=device)
