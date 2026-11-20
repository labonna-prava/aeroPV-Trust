"""
Early Fusion Multimodal PV Inspection Model
Author: Labonna Prava
"""

import torch
import torch.nn as nn
from .backbones import make_resnet18


class EarlyFusion(nn.Module):
    """
    Early Fusion multimodal model:
    Concatenates RGB (3 channels) and Thermal (1 channel) into a 4-channel tensor
    before feature extraction with a shared ResNet-18 backbone.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = False):
        super().__init__()
        self.backbone = make_resnet18(in_channels=4, pretrained=pretrained)
        self.head = nn.Linear(512, num_classes)

    def forward(self, rgb: torch.Tensor, th: torch.Tensor) -> torch.Tensor:
        # Concatenate along channel dimension: [B, 4, H, W]
        x = torch.cat([rgb, th], dim=1)
        feat = self.backbone(x)
        return self.head(feat)

    def extract_features(self, rgb: torch.Tensor, th: torch.Tensor) -> torch.Tensor:
        x = torch.cat([rgb, th], dim=1)
        return self.backbone(x)
