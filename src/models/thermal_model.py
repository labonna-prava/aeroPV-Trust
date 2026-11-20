"""
Unimodal Thermal-Only PV Inspection Model
Author: Labonna Prava
"""

import torch
import torch.nn as nn
from .backbones import make_resnet18


class ThermalOnly(nn.Module):
    """
    Unimodal Thermal classifier using a 1-channel ResNet-18 backbone.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = False):
        super().__init__()
        self.backbone = make_resnet18(in_channels=1, pretrained=pretrained)
        self.head = nn.Linear(512, num_classes)

    def forward(self, rgb: torch.Tensor = None, th: torch.Tensor = None) -> torch.Tensor:
        if th is None and rgb is not None:
            th = rgb  # fallback if passed as single arg
        feat = self.backbone(th)
        return self.head(feat)

    def extract_features(self, th: torch.Tensor) -> torch.Tensor:
        return self.backbone(th)
