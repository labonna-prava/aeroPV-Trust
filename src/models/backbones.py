"""
Backbone architectures and ResNet builders
Author: Labonna Prava
"""

import torch
import torch.nn as nn
import torchvision


def make_resnet18(in_channels: int = 3, pretrained: bool = False) -> nn.Module:
    """
    Construct a modified ResNet-18 backbone.
    Adjusts first convolution layer for custom channel count (e.g. 1-ch thermal, 4-ch early fusion),
    and removes final classification head for direct feature representation.
    """
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = torchvision.models.resnet18(weights=weights)

    if in_channels != 3:
        old_conv = model.conv1
        model.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        if pretrained and in_channels == 1:
            # Average RGB channel weights for single-channel thermal initialization
            with torch.no_grad():
                model.conv1.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))

    model.fc = nn.Identity()
    return model
