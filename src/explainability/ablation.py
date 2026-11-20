"""
Modality Ablation and Global Contribution Analysis
Author: Labonna Prava
"""

from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ..config import DEVICE


@torch.no_grad()
def evaluate_modality_ablation(
    late_model: torch.nn.Module,
    dataloader: DataLoader,
    device: str = DEVICE,
) -> Dict[str, float]:
    """
    Ablate modalities by zeroing RGB or Thermal representations in Late Fusion.
    Quantifies average prediction probability drop:
    - delta_rgb = P(full) - P(without RGB)
    - delta_th  = P(full) - P(without Thermal)
    """
    late_model.eval().to(device)

    p_full_list = []
    p_no_rgb_list = []
    p_no_th_list = []

    for rgb, th, _ in dataloader:
        rgb = rgb.to(device)
        th = th.to(device)

        zeros_rgb = torch.zeros_like(rgb)
        zeros_th = torch.zeros_like(th)

        p_full = F.softmax(late_model(rgb, th), dim=1)[:, 1].cpu().numpy()
        p_no_rgb = F.softmax(late_model(zeros_rgb, th), dim=1)[:, 1].cpu().numpy()
        p_no_th = F.softmax(late_model(rgb, zeros_th), dim=1)[:, 1].cpu().numpy()

        p_full_list.extend(p_full)
        p_no_rgb_list.extend(p_no_rgb)
        p_no_th_list.extend(p_no_th)

    p_full_arr = np.asarray(p_full_list)
    p_no_rgb_arr = np.asarray(p_no_rgb_list)
    p_no_th_arr = np.asarray(p_no_th_list)

    drop_rgb = p_full_arr - p_no_rgb_arr
    drop_th = p_full_arr - p_no_th_arr

    return {
        "mean_drop_rgb_removed": float(np.mean(drop_rgb)),
        "mean_drop_thermal_removed": float(np.mean(drop_th)),
        "std_drop_rgb": float(np.std(drop_rgb)),
        "std_drop_thermal": float(np.std(drop_th)),
        "thermal_dominance_ratio": float(np.mean(drop_th > drop_rgb)),
    }


@torch.no_grad()
def compute_modality_contributions(
    late_model: torch.nn.Module,
    rgb_tensor: torch.Tensor,
    th_tensor: torch.Tensor,
    device: str = DEVICE,
) -> Tuple[float, float, float]:
    """Compute individual sample contribution deltas for (RGB, Thermal)."""
    late_model.eval().to(device)
    rgb = rgb_tensor.unsqueeze(0).to(device)
    th = th_tensor.unsqueeze(0).to(device)

    p_full = float(F.softmax(late_model(rgb, th), dim=1)[0, 1].item())
    p_no_rgb = float(F.softmax(late_model(torch.zeros_like(rgb), th), dim=1)[0, 1].item())
    p_no_th = float(F.softmax(late_model(rgb, torch.zeros_like(th)), dim=1)[0, 1].item())

    delta_rgb = p_full - p_no_rgb
    delta_th = p_full - p_no_th
    return p_full, delta_rgb, delta_th
