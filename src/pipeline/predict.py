"""
End-to-End UAV PV Inspection Pipeline
Author: Labonna Prava
"""

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms

from ..config import DEVICE, IMAGE_SIZE, IR_CROP, RGB_CROP
from ..data.thermogram import Thermogram, load_rgb, load_thermal
from ..data.segmentation import (
    crop_image,
    make_aligned_pair,
    extract_module_crops,
)
from ..models.late_fusion import LateFusion
from ..explainability.gradcam import GradCAM, generate_cam_overlay


class PVInspectionPipeline:
    """
    High-level automated inspection pipeline:
    1. Loads paired UAV RGB & radiometric Thermal images
    2. Aligns scenes and crops inspection ROI
    3. Segments candidate PV modules using HSV Otsu & connected components
    4. Evaluates each module with Late Fusion deep learning model
    5. Computes multi-level Grad-CAM explanations and generates summary reports
    """

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        device: str = DEVICE,
    ):
        self.device = device
        self.model = LateFusion(num_classes=2, pretrained=False).to(device)

        if checkpoint_path is not None and Path(checkpoint_path).exists():
            state_dict = torch.load(checkpoint_path, map_location=device)
            self.model.load_state_dict(state_dict)

        self.model.eval()

        self.rgb_tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        self.th_tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.25,)),
            ]
        )

    def inspect_pair(
        self,
        rgb_source: Union[str, Path, np.ndarray],
        thermal_source: Union[str, Path, Thermogram, np.ndarray],
        anomaly_threshold: float = 0.5,
        generate_heatmaps: bool = True,
    ) -> Dict:
        """
        Run end-to-end inspection on an RGB and Thermal UAV image pair.
        """
        # Load RGB
        if isinstance(rgb_source, (str, Path)):
            rgb_raw = load_rgb(Path(rgb_source))
        else:
            rgb_raw = rgb_source

        # Load Thermal
        if isinstance(thermal_source, (str, Path)):
            th_obj = load_thermal(Path(thermal_source))
            th_celsius = th_obj.celsius
            th8 = th_obj.render8()
        elif isinstance(thermal_source, Thermogram):
            th_celsius = thermal_source.celsius
            th8 = thermal_source.render8()
        else:
            th8 = thermal_source
            th_celsius = thermal_source.astype(np.float32)

        # Align pair
        rgb_aligned, th8_aligned = make_aligned_pair(rgb_raw, th8)
        thC_crop = crop_image(th_celsius, IR_CROP)

        # Extract module crops
        samples, mask01, labels = extract_module_crops(rgb_aligned, th8_aligned, thC_crop)

        annotated_rgb = rgb_aligned.copy()
        annotated_thermal = cv2.applyColorMap(th8_aligned, cv2.COLORMAP_INFERNO)

        modules_info = []

        for idx, sample in enumerate(samples):
            rgb_p = sample["rgb_patch"]
            th8_p = sample["th8_patch"]

            # Prepare tensors
            rgb_tensor = self.rgb_tf(rgb_p[:, :, ::-1].copy()).unsqueeze(0).to(self.device)
            th_tensor = self.th_tf(th8_p[:, :, None] if th8_p.ndim == 2 else th8_p).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(rgb_tensor, th_tensor)
                prob = float(F.softmax(logits, dim=1)[0, 1].item())
                pred = int(prob >= anomaly_threshold)

            bbox = sample["bbox"]
            x1, y1, x2, y2 = bbox

            # Grad-CAM if anomalous and requested
            rgb_overlay = None
            th_overlay = None
            if generate_heatmaps:
                try:
                    rgb_layer = self.model.rgb_net.layer4[-1].conv2
                    with GradCAM(self.model, rgb_layer) as gc:
                        cam_rgb, _, _ = gc(lambda: self.model(rgb_tensor, th_tensor), class_idx=1)
                        rgb_overlay = generate_cam_overlay(rgb_p, cam_rgb)

                    th_layer = self.model.th_net.layer4[-1].conv2
                    with GradCAM(self.model, th_layer) as gc:
                        cam_th, _, _ = gc(lambda: self.model(rgb_tensor, th_tensor), class_idx=1)
                        th_overlay = generate_cam_overlay(th8_p, cam_th)
                except Exception:
                    pass

            # Annotate full scene images
            box_color = (0, 0, 255) if pred == 1 else (0, 255, 0)
            cv2.rectangle(annotated_rgb, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(
                annotated_rgb,
                f"M{idx:02d}: {prob:.2f}",
                (x1, max(y1 - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                box_color,
                1,
            )

            cv2.rectangle(annotated_thermal, (x1, y1), (x2, y2), box_color, 2)

            modules_info.append(
                {
                    "module_index": idx,
                    "bbox": bbox,
                    "anomaly_probability": prob,
                    "is_defect": bool(pred),
                    "thermal_score": float(sample["score"]),
                    "rgb_patch": rgb_p,
                    "th8_patch": th8_p,
                    "rgb_cam_overlay": rgb_overlay,
                    "th_cam_overlay": th_overlay,
                }
            )

        num_defects = sum(1 for m in modules_info if m["is_defect"])

        return {
            "total_modules_detected": len(modules_info),
            "anomalies_detected": num_defects,
            "defect_rate": float(num_defects / len(modules_info)) if modules_info else 0.0,
            "annotated_rgb": annotated_rgb,
            "annotated_thermal": annotated_thermal,
            "segmentation_mask": mask01,
            "modules": modules_info,
        }
