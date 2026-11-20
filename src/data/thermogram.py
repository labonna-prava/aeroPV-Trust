"""
Thermogram and Radiometric Image Processing
Author: Labonna Prava
"""

from pathlib import Path
import numpy as np
import cv2
import tifffile as tif


class Thermogram:
    """
    Handles radiometric thermal TIFF images from UAV cameras.
    Converts raw 16-bit sensor counts to calibrated degrees Celsius and 8-bit visual maps.
    Formula: Celsius = (raw_counts * 0.04) - 273.15
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Thermogram file not found: {self.path}")

    @property
    def raw(self) -> np.ndarray:
        return tif.imread(self.path.as_posix())

    @property
    def celsius(self) -> np.ndarray:
        raw_val = self.raw.astype(np.float32)
        return (raw_val * 0.04) - 273.15

    def render8(self) -> np.ndarray:
        """Normalized 8-bit representation for display and visual modeling."""
        c = self.celsius.astype(np.float32)
        c_min, c_max = c.min(), c.max()
        if c_max - c_min < 1e-5:
            return np.zeros_like(c, dtype=np.uint8)
        norm = (c - c_min) / (c_max - c_min)
        return (norm * 255.0).astype(np.uint8)

    def colormap(self, cmap_code: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
        """Convert thermal 8-bit image to false-color RGB for visual inspection."""
        r8 = self.render8()
        return cv2.applyColorMap(r8, cmap_code)


def load_rgb(path: Path) -> np.ndarray:
    """Load an RGB image in BGR format."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"RGB image not found: {p}")
    img = cv2.imread(p.as_posix(), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image at: {p}")
    return img


def load_thermal(path: Path) -> Thermogram:
    """Load a radiometric thermal TIFF image wrapped in Thermogram object."""
    return Thermogram(path)
