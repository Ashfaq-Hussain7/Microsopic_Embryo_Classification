# model/pretrained_model.py
import torch.nn as nn
from torchvision import models

def get_pretrained_model(name="resnet18", num_classes=2):
    if name == "resnet18":
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "vgg16":
        model = models.vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(4096, num_classes)
    return model
