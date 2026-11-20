"""
Late Fusion Multimodal PV Inspection Model
Author: Labonna Prava
"""

from typing import Tuple
import torch
import torch.nn as nn
from .backbones import make_resnet18


class LateFusion(nn.Module):
    """
    Late Fusion multimodal architecture:
    Processes RGB and Thermal modalities with separate ResNet-18 encoders.
    Each branch is projected to a 256-dimensional representation, concatenated
    into a 512-dimensional fused representation, and classified via a shared head.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = False):
        super().__init__()
        self.rgb_net = make_resnet18(in_channels=3, pretrained=pretrained)
        self.th_net = make_resnet18(in_channels=1, pretrained=pretrained)

        self.proj_rgb = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.proj_th = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.head = nn.Linear(512, num_classes)

    def get_embeddings(self, rgb: torch.Tensor, th: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract modality-specific embeddings and fused embedding vector:
        Returns (z_rgb, z_th, z_fused)
        """
        f_rgb = self.rgb_net(rgb)
        f_th = self.th_net(th)

        z_rgb = self.proj_rgb(f_rgb)
        z_th = self.proj_th(f_th)
        z_fused = torch.cat([z_rgb, z_th], dim=1)
        return z_rgb, z_th, z_fused

    def forward(self, rgb: torch.Tensor, th: torch.Tensor) -> torch.Tensor:
        _, _, z_fused = self.get_embeddings(rgb, th)
        return self.head(z_fused)

    def classify_embedding(self, z_fused: torch.Tensor) -> torch.Tensor:
        """Classify directly from a 512-dimensional fused embedding vector."""
        return self.head(z_fused)
