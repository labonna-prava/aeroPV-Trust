"""
PV Module Segmentation, Image Alignment, and Anomaly Scoring
Author: Labonna Prava
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
from scipy import ndimage as ndi
from ..config import IR_CROP, RGB_CROP, MAD_EPS


def crop_image(arr: np.ndarray, box: Dict[str, int]) -> np.ndarray:
    """Crop an image array using left, top, right, bottom boundaries."""
    l, t, r, b = box["left"], box["top"], box["right"], box["bottom"]
    return arr[t:b, l:r]


def make_aligned_pair(
    rgb_bgr: np.ndarray,
    thermal8: np.ndarray,
    rgb_box: Optional[Dict[str, int]] = None,
    ir_box: Optional[Dict[str, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Spatially align RGB and Thermal frames using calibrated bounding crops.
    Resizes the RGB crop to match the spatial resolution of the Thermal crop.
    """
    if rgb_box is None:
        rgb_box = RGB_CROP
    if ir_box is None:
        ir_box = IR_CROP

    th_crop = crop_image(thermal8, ir_box)
    rgb_crop = crop_image(rgb_bgr, rgb_box)

    th_h, th_w = th_crop.shape[:2]
    rgb_crop_aligned = cv2.resize(rgb_crop, (th_w, th_h), interpolation=cv2.INTER_AREA)
    return rgb_crop_aligned, th_crop


def preprocess_rgb_image_hsv_otsu_bgr(bgr: np.ndarray) -> np.ndarray:
    """
    Extract PV module candidate regions using multi-channel HSV Otsu thresholding.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    masks = []
    for ci, ch in enumerate(cv2.split(hsv)):
        ch_blurred = cv2.GaussianBlur(ch, (3, 3), 0)
        max_val = 179 if ci == 0 else 255
        _, mask = cv2.threshold(
            ch_blurred, 0, max_val, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        masks.append(mask)

    m = cv2.bitwise_and(masks[0], masks[1])
    m = cv2.bitwise_and(m, masks[2])
    return m


def postprocess_binary_mask(mask: np.ndarray) -> np.ndarray:
    """
    Clean the binary segmentation mask using median blurring,
    morphological opening, and hole filling.
    """
    mask = cv2.medianBlur(mask, 5)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    filled = ndi.binary_fill_holes(mask > 0)
    return filled.astype(np.uint8)


def connected_components(mask01: np.ndarray) -> Tuple[int, np.ndarray]:
    """Perform connected component labeling on binary mask."""
    n, labels = cv2.connectedComponents(mask01.astype(np.uint8))
    return n, labels


def robust_mad(x: np.ndarray, eps: float = MAD_EPS) -> float:
    """Compute the Median Absolute Deviation (MAD) for robust anomaly scaling."""
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + eps
    return float(mad)


def bbox_from_label(labels: np.ndarray, label_id: int) -> Optional[Tuple[int, int, int, int]]:
    """Extract (x1, y1, x2, y2) bounding box from a labeled connected component."""
    ys, xs = np.where(labels == label_id)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def extract_module_crops(
    rgb_crop_bgr: np.ndarray,
    thermal8_crop: np.ndarray,
    thermal_celsius_crop: np.ndarray,
    min_area: int = 1500,
    max_area: int = 120000,
) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
    """
    Segment PV modules and extract paired RGB and Thermal crops along with
    their robust thermal anomaly score:
        score = (T95(module) - median(scene)) / MAD(scene)
    """
    mask_raw = preprocess_rgb_image_hsv_otsu_bgr(rgb_crop_bgr)
    mask01 = postprocess_binary_mask(mask_raw)
    n_labels, labels = connected_components(mask01)

    img_med = float(np.median(thermal_celsius_crop))
    img_mad = robust_mad(thermal_celsius_crop)

    samples = []
    for label_id in range(1, n_labels):
        bbox = bbox_from_label(labels, label_id)
        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        area = w * h

        if area < min_area or area > max_area:
            continue

        rgb_patch = rgb_crop_bgr[y1:y2, x1:x2]
        th8_patch = thermal8_crop[y1:y2, x1:x2]
        thC_patch = thermal_celsius_crop[y1:y2, x1:x2]

        t95 = float(np.percentile(thC_patch, 95))
        score = (t95 - img_med) / img_mad

        # Default pseudo-label threshold
        label = int(score > 2.2931)

        samples.append(
            {
                "rgb_patch": rgb_patch,
                "th8_patch": th8_patch,
                "thC_patch": thC_patch,
                "label": label,
                "score": score,
                "bbox": (x1, y1, x2, y2),
                "area": area,
            }
        )

    return samples, mask01, labels
