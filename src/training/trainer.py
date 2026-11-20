"""
Model Trainer and Validation Routines
Author: Labonna Prava
"""

from typing import Dict, Optional, Tuple
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from ..config import DEVICE, LEARNING_RATE, EPOCHS, WEIGHT_DECAY
from .metrics import compute_classification_metrics


def forward_pass(model: nn.Module, rgb: torch.Tensor, th: torch.Tensor) -> torch.Tensor:
    """Dispatches input tensors based on model architecture requirements."""
    name = model.__class__.__name__
    if name == "RGBOnly":
        return model(rgb)
    elif name == "ThermalOnly":
        return model(None, th)
    else:
        return model(rgb, th)


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer] = None,
    class_weights: Optional[torch.Tensor] = None,
    device: str = DEVICE,
) -> Dict[str, float]:
    """Execute a single training or evaluation epoch."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    all_y_true = []
    all_y_probs = []

    if class_weights is not None:
        class_weights = class_weights.to(device)

    for rgb, th, y in dataloader:
        rgb = rgb.to(device)
        th = th.to(device)
        y = y.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = forward_pass(model, rgb, th)
            loss = F.cross_entropy(logits, y, weight=class_weights)

            if is_train:
                loss.backward()
                optimizer.step()

        probs = F.softmax(logits, dim=1)[:, 1]

        total_loss += loss.item() * len(y)
        all_y_true.extend(y.detach().cpu().numpy())
        all_y_probs.extend(probs.detach().cpu().numpy())

    epoch_loss = total_loss / max(len(all_y_true), 1)
    metrics = compute_classification_metrics(all_y_true, all_y_probs)
    metrics["loss"] = epoch_loss

    return metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    class_weights: Optional[torch.Tensor] = None,
    device: str = DEVICE,
    save_path: Optional[Path] = None,
    verbose: bool = True,
) -> Tuple[nn.Module, Dict[str, list]]:
    """
    Train a PV anomaly model with class-imbalanced cross-entropy loss
    and track evaluation metrics across epochs.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {
        "train_loss": [],
        "train_auc": [],
        "val_loss": [],
        "val_auc": [],
        "val_ap": [],
        "val_f1": [],
    }

    best_val_auc = 0.0

    for epoch in range(1, epochs + 1):
        train_m = run_epoch(
            model, train_loader, optimizer=optimizer, class_weights=class_weights, device=device
        )
        history["train_loss"].append(train_m["loss"])
        history["train_auc"].append(train_m["roc_auc"])

        val_str = ""
        if val_loader is not None:
            val_m = evaluate_model(model, val_loader, class_weights=class_weights, device=device)
            history["val_loss"].append(val_m["loss"])
            history["val_auc"].append(val_m["roc_auc"])
            history["val_ap"].append(val_m["ap"])
            history["val_f1"].append(val_m["f1"])

            val_str = f" | Val Loss: {val_m['loss']:.4f} | Val AUC: {val_m['roc_auc']:.3f} | Val AP: {val_m['ap']:.3f} | Val F1: {val_m['f1']:.3f}"

            if val_m["roc_auc"] > best_val_auc and save_path is not None:
                best_val_auc = val_m["roc_auc"]
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_path)

        if verbose:
            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] "
                f"Train Loss: {train_m['loss']:.4f} | Train AUC: {train_m['roc_auc']:.3f}"
                + val_str
            )

    # If save_path provided and not saved yet, save final model state
    if save_path is not None and not save_path.exists():
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)

    return model, history


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    class_weights: Optional[torch.Tensor] = None,
    device: str = DEVICE,
) -> Dict[str, float]:
    """Evaluate model on a dataloader without computing gradients."""
    return run_epoch(
        model, dataloader, optimizer=None, class_weights=class_weights, device=device
    )


@torch.no_grad()
def predict_dataloader(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = DEVICE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate probability predictions and ground truth labels from a dataloader."""
    model.eval().to(device)
    ys, ps = [], []
    for rgb, th, y in dataloader:
        rgb = rgb.to(device)
        th = th.to(device)
        logits = forward_pass(model, rgb, th)
        probs = F.softmax(logits, dim=1)[:, 1]

        ys.extend(y.cpu().numpy())
        ps.extend(probs.cpu().numpy())

    return np.asarray(ys), np.asarray(ps)
