import torchmetrics


def get_wandb_config():
    config = {
        "architecture": "CNN",
        "dataset": "CIFAR-10",
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 1,
    }
    return config


def get_metrics_config():
    accuracy_metric = torchmetrics.Accuracy(task='multiclass', num_classes=10)
    precision_metric = torchmetrics.Precision(task='multiclass', num_classes=10)
    recall_metric = torchmetrics.Recall(task='multiclass', num_classes=10)
    f1_metric = torchmetrics.F1Score(task='multiclass', num_classes=10)
    return accuracy_metric, precision_metric, recall_metric, f1_metric
