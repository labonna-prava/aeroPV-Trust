"""
Multi-Level Grad-CAM Attribution for Unimodal and Multimodal PV Models
Author: Labonna Prava
"""

from typing import Callable, Optional, Tuple
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) context manager.
    Hooks into targeted convolutional layers to capture feature activations and gradients.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.hook_handles = []

    def __enter__(self):
        def forward_hook(module, inp, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self.hook_handles:
            h.remove()
        self.hook_handles.clear()

    def __call__(
        self,
        forward_fn: Callable[[], torch.Tensor],
        class_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """
        Executes backward pass and returns normalized 2D Grad-CAM heatmap,
        predicted class index, and predicted class probability.
        """
        self.model.zero_grad()
        logits = forward_fn()

        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())

        score = logits[0, class_idx]
        score.backward(retain_graph=True)

        prob = float(F.softmax(logits, dim=1)[0, class_idx].item())

        grads = self.gradients  # [1, C, H, W]
        acts = self.activations  # [1, C, H, W]

        # Global average pooling of gradients as channel importance weights
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * acts, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Spatial normalization to [0, 1]
        cam = cam.squeeze().detach().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-6:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, class_idx, prob


def generate_cam_overlay(
    image: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Resize 2D CAM heatmap to image dimensions and blend as a color heatmap overlay.
    """
    h, w = image.shape[:2]
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    cam_uint8 = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(cam_uint8, colormap)

    if image.ndim == 2:
        img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 1:
        img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = image.copy()

    overlay = cv2.addWeighted(img_bgr, 1.0 - alpha, heatmap, alpha, 0)
    return overlay
