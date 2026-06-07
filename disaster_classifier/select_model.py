import torch.nn as nn
import torchvision.models as models

def get_model(name: str, num_classes: int):
    """
    Returns a pre-trained model with the final classification layer modified.
    Args:
        name (str): Model name. Options: 'resnet18', 'resnet34', 'resnet50', 'mobilenet_v2', 'efficientnet_b0'
        num_classes (int): Number of output classes.
    """
    name = name.lower()

    if name == "resnet18":
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif name == "resnet34":
        model = models.resnet34(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif name == "resnet50":
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    else:
        raise ValueError(f"Model '{name}' not supported.")

    return model
