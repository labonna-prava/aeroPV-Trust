"""
Cross-Modal Counterfactual Multimodal Evaluation (CCME)
Author: Labonna Prava
"""

from typing import Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ..config import DEVICE


class CrossModalCounterfactualExplainer:
    """
    Evaluates cross-modal counterfactual dependencies in the multimodal latent space.
    Calculates minimum interpolation factor alpha in [0, 1] needed along each modality
    branch to flip the prediction from normal to anomalous.
    """

    def __init__(self, late_model: torch.nn.Module, device: str = DEVICE):
        self.model = late_model.eval().to(device)
        self.device = device
        self.base_z_rgb: Optional[torch.Tensor] = None
        self.base_z_th: Optional[torch.Tensor] = None

    @torch.no_grad()
    def fit_normal_baseline(self, dataloader: DataLoader):
        """Compute the average normal (label 0) embedding for RGB and Thermal."""
        self.model.eval()
        rgb_list, th_list = [], []

        for rgb, th, y in dataloader:
            normal_mask = (y == 0)
            if not normal_mask.any():
                continue

            rgb_n = rgb[normal_mask].to(self.device)
            th_n = th[normal_mask].to(self.device)

            z_rgb, z_th, _ = self.model.get_embeddings(rgb_n, th_n)
            rgb_list.append(z_rgb.cpu())
            th_list.append(z_th.cpu())

        all_rgb = torch.cat(rgb_list, dim=0)
        all_th = torch.cat(th_list, dim=0)

        self.base_z_rgb = all_rgb.mean(dim=0, keepdim=True).to(self.device)
        self.base_z_th = all_th.mean(dim=0, keepdim=True).to(self.device)

    @torch.no_grad()
    def explain(
        self,
        rgb_tensor: torch.Tensor,
        th_tensor: torch.Tensor,
        decision_threshold: float = 0.5,
        steps: int = 25,
    ) -> Dict[str, float]:
        """
        Binary search for minimum alpha_rgb and alpha_th required to trigger anomaly decision.
        """
        if self.base_z_rgb is None or self.base_z_th is None:
            raise RuntimeError("Baseline not computed. Call fit_normal_baseline() first.")

        rgb = rgb_tensor.unsqueeze(0).to(self.device)
        th = th_tensor.unsqueeze(0).to(self.device)

        z_sample_rgb, z_sample_th, _ = self.model.get_embeddings(rgb, th)

        # 1. Thermal fixed to baseline, vary RGB from 0.0 to 1.0
        low_rgb, high_rgb = 0.0, 1.0
        alpha_rgb = 1.0
        for _ in range(steps):
            mid = (low_rgb + high_rgb) / 2.0
            z_interp_rgb = mid * z_sample_rgb + (1.0 - mid) * self.base_z_rgb
            z_fused = torch.cat([z_interp_rgb, self.base_z_th], dim=1)
            p = F.softmax(self.model.classify_embedding(z_fused), dim=1)[0, 1].item()
            if p >= decision_threshold:
                alpha_rgb = mid
                high_rgb = mid
            else:
                low_rgb = mid

        # 2. RGB fixed to baseline, vary Thermal from 0.0 to 1.0
        low_th, high_th = 0.0, 1.0
        alpha_th = 1.0
        for _ in range(steps):
            mid = (low_th + high_th) / 2.0
            z_interp_th = mid * z_sample_th + (1.0 - mid) * self.base_z_th
            z_fused = torch.cat([self.base_z_rgb, z_interp_th], dim=1)
            p = F.softmax(self.model.classify_embedding(z_fused), dim=1)[0, 1].item()
            if p >= decision_threshold:
                alpha_th = mid
                high_th = mid
            else:
                low_th = mid

        full_prob = F.softmax(self.model(rgb, th), dim=1)[0, 1].item()

        return {
            "predicted_prob": float(full_prob),
            "alpha_rgb_min": float(alpha_rgb),
            "alpha_thermal_min": float(alpha_th),
            "thermal_faster_transition": bool(alpha_th < alpha_rgb),
        }
