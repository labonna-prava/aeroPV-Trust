"""
Evaluation and Reliability Metrics for PV Anomaly Inspection
Author: Labonna Prava
"""

from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    brier_score_loss,
    roc_curve,
    precision_recall_curve,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute comprehensive metrics for class-imbalanced anomaly detection:
    ROC-AUC, Average Precision (AP), F1-score, Balanced Accuracy, Brier score.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {}

    # ROC-AUC (requires at least one sample from each class)
    try:
        if len(np.unique(y_true)) > 1:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        else:
            metrics["roc_auc"] = 0.5
    except Exception:
        metrics["roc_auc"] = 0.5

    # Average Precision
    try:
        if len(np.unique(y_true)) > 1:
            metrics["ap"] = float(average_precision_score(y_true, y_prob))
        else:
            metrics["ap"] = float(np.mean(y_true))
    except Exception:
        metrics["ap"] = 0.0

    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics["balanced_acc"] = float(balanced_accuracy_score(y_true, y_pred))
    metrics["brier_score"] = float(brier_score_loss(y_true, y_prob))

    # Confusion matrix elements
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["tn"] = int(tn)
    metrics["fp"] = int(fp)
    metrics["fn"] = int(fn)
    metrics["tp"] = int(tp)
    metrics["precision"] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    metrics["recall"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    return metrics


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Brier reliability score: E[(p - y)^2]. Lower is more calibrated."""
    return float(brier_score_loss(y_true, y_prob))
