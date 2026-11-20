"""
Data loading and preprocessing utilities for AeroPV-Trust.
Author: Labonna Prava
"""

from .thermogram import Thermogram, load_rgb, load_thermal
from .segmentation import (
    crop_image,
    preprocess_rgb_image_hsv_otsu_bgr,
    postprocess_binary_mask,
    connected_components,
    extract_module_crops,
    robust_mad,
)
from .dataset import PairedModuleDataset, create_dataloaders
from .generator import generate_synthetic_uav_dataset

__all__ = [
    "Thermogram",
    "load_rgb",
    "load_thermal",
    "crop_image",
    "preprocess_rgb_image_hsv_otsu_bgr",
    "postprocess_binary_mask",
    "connected_components",
    "extract_module_crops",
    "robust_mad",
    "PairedModuleDataset",
    "create_dataloaders",
    "generate_synthetic_uav_dataset",
]
