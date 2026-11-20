"""
CLI Script: Model Evaluation and Performance Verification
Author: Labonna Prava
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch

from src.config import CROPS_DIR, CHECKPOINTS_DIR, RESULTS_DIR, DEVICE
from src.data.dataset import create_dataloaders
from src.models.rgb_model import RGBOnly
from src.models.thermal_model import ThermalOnly
from src.models.early_fusion import EarlyFusion
from src.models.late_fusion import LateFusion
from src.training.trainer import evaluate_model


def main():
    index_csv = CROPS_DIR / "index.csv"
    if not index_csv.exists():
        print(f"Error: {index_csv} not found.")
        return

    _, test_loader, _, test_df = create_dataloaders(index_csv=index_csv, batch_size=16)

    model_configs = [
        ("RGB-only", RGBOnly(num_classes=2), CHECKPOINTS_DIR / "rgb_only.pt"),
        ("Thermal-only", ThermalOnly(num_classes=2), CHECKPOINTS_DIR / "thermal_only.pt"),
        ("Early Fusion", EarlyFusion(num_classes=2), CHECKPOINTS_DIR / "early_fusion.pt"),
        ("Late Fusion", LateFusion(num_classes=2), CHECKPOINTS_DIR / "late_fusion.pt"),
    ]

    records = []
    print(f"\n[AeroPV-Trust] Evaluating models on {len(test_df)} test samples:\n")
    for name, model, ckpt in model_configs:
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            model.eval()

        m = evaluate_model(model, test_loader, device=DEVICE)
        records.append(
            {
                "Model": name,
                "ROC-AUC": round(m["roc_auc"], 3),
                "AP": round(m["ap"], 3),
                "F1-score": round(m["f1"], 3),
                "Balanced Accuracy": round(m["balanced_acc"], 3),
                "Brier Score": round(m["brier_score"], 4),
            }
        )

    eval_df = pd.DataFrame(records)
    print(eval_df.to_string(index=False))

    out_file = RESULTS_DIR / "evaluation_summary.csv"
    eval_df.to_csv(out_file, index=False)
    print(f"\nSaved evaluation summary to: {out_file}")


if __name__ == "__main__":
    main()
