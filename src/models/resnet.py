from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet18


def build_resnet18(num_classes: int = 100) -> nn.Module:
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

