"""
Model architectures for multimodal PV inspection.
Author: Labonna Prava
"""

from .backbones import make_resnet18
from .rgb_model import RGBOnly
from .thermal_model import ThermalOnly
from .early_fusion import EarlyFusion
from .late_fusion import LateFusion

__all__ = [
    "make_resnet18",
    "RGBOnly",
    "ThermalOnly",
    "EarlyFusion",
    "LateFusion",
]
