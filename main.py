import torch.optim as optim
from torchvision.models import ResNet18_Weights, resnet18
import wandb
from gradcam import run_gradcam
from setup.config import get_wandb_config, get_device
from setup.base_model import BaseCNN
import torch.nn as nn
from setup.data import get_data, get_labels
from train import set_parameter_requires_grad, train_model
from sklearn.model_selection import KFold
import torch
from torch.utils.data import Subset

# Settings for model and fine-tuning
dropout = 0.5
learning_rate = 0.001
momentum = 0.9
batch_size = 32
epochs = 5
augment = True
n_splits = 5
model_complexity = 'large'
optimizer = 'sgd'
model_name = 'resnet-18'
version = 'gradcam with resize'
scheduler = ''
feature_extract = False

if __name__ == '__main__':
    all_fold_accuracies = []
    # Set up W&B
    wandb_config = get_wandb_config(model_name, model_complexity, learning_rate, betas=None, epsilon=None, conv_layers=None, linear_layers=None,
                                    pooling=None, batch_norm=None, dropout=None, L2=None, weights_init=None, data_augmentation=augment,
                                    optimizer=optimizer, scheduler=None, batch_size=batch_size)

    wandb.init(project="CIFAR-10-Classification", config=wandb_config, name=version, notes='Cross-validation experiment')

    # Set up dataset
    traindata, testdata, trainloader, testloader = get_data(batch_size)
    classes = get_labels()

    # KFold Cross-Validation setup
    kfold = KFold(n_splits=n_splits, shuffle=True)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(traindata)):
        print(f'Fold {fold+1}/{n_splits}')

        # Subset the dataset for training and validation using indices from KFold
        train_subset = Subset(traindata, train_idx)
        val_subset = Subset(traindata, val_idx)

        trainloader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        valloader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        dataloaders_dict = {'training': trainloader, 'validation': valloader}

        # Set up model
        if model_name == 'base':
            model = BaseCNN()
        elif model_name == 'resnet-18':
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

        # Train and validate model for this fold
        print(f'Starting training for Fold {fold+1}')
        trained_model, best_acc = train_model(model, dataloaders_dict, criterion, optimizer, classes, num_epochs=epochs, device=device, fold=fold+1)

        # Log GradCAM results for this fold
        run_gradcam(trained_model, testloader, device, classes)

        # Log fold results to W&B
        wandb.log({"fold": fold + 1})

        all_fold_accuracies.append(best_acc)

    # Mean and standard deviation after all folds
    mean_accuracy = sum(all_fold_accuracies) / len(all_fold_accuracies)
    std_accuracy = (sum((x - mean_accuracy) ** 2 for x in all_fold_accuracies) / len(all_fold_accuracies)) ** 0.5

    wandb.log({
        "mean_accuracy": mean_accuracy,
        "std_accuracy": std_accuracy
    })
