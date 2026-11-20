"""
Unimodal RGB-Only PV Inspection Model
Author: Labonna Prava
"""

import torch
import torch.nn as nn
from .backbones import make_resnet18


class RGBOnly(nn.Module):
    """
    Unimodal RGB classifier using a 3-channel ResNet-18 backbone.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = False):
        super().__init__()
        self.backbone = make_resnet18(in_channels=3, pretrained=pretrained)
        self.head = nn.Linear(512, num_classes)

    def forward(self, rgb: torch.Tensor, th: torch.Tensor = None) -> torch.Tensor:
        feat = self.backbone(rgb)
        return self.head(feat)

    def extract_features(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.backbone(rgb)
