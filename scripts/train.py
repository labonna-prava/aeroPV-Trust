"""
CLI Script: Train All 4 Inspection Models (RGB, Thermal, Early Fusion, Late Fusion)
Author: Labonna Prava
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch

from src.config import CROPS_DIR, CHECKPOINTS_DIR, RESULTS_DIR, DEVICE, EPOCHS, BATCH_SIZE, LEARNING_RATE
from src.data.dataset import create_dataloaders
from src.models.rgb_model import RGBOnly
from src.models.thermal_model import ThermalOnly
from src.models.early_fusion import EarlyFusion
from src.models.late_fusion import LateFusion
from src.training.trainer import train_model, evaluate_model


def main():
    parser = argparse.ArgumentParser(description="Train PV anomaly detection models")
    parser.add_argument("--index_csv", type=str, default=str(CROPS_DIR / "index.csv"))
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    index_csv = Path(args.index_csv)
    if not index_csv.exists():
        raise FileNotFoundError(f"Index CSV not found at: {index_csv}. Run extract_crops.py first.")

    print(f"[AeroPV-Trust] Loading dataset from: {index_csv}")
    train_loader, test_loader, train_df, test_df = create_dataloaders(
        index_csv=index_csv,
        batch_size=args.batch_size,
    )

    print(f"[AeroPV-Trust] Training samples: {len(train_df)}, Testing samples: {len(test_df)}")

    # Class weights for imbalanced cross-entropy
    counts = train_df["label"].value_counts().to_dict()
    n0, n1 = counts.get(0, 1), counts.get(1, 1)
    w0 = 1.0 / n0
    w1 = 1.0 / n1
    weights = torch.tensor([w0, w1], dtype=torch.float32)
    weights = weights / weights.sum()
    print(f"[AeroPV-Trust] Normalized class weights: {weights.tolist()} on device {DEVICE}")

    models = {
        "RGB-only": (RGBOnly(num_classes=2, pretrained=False), CHECKPOINTS_DIR / "rgb_only.pt"),
        "Thermal-only": (ThermalOnly(num_classes=2, pretrained=False), CHECKPOINTS_DIR / "thermal_only.pt"),
        "Early Fusion": (EarlyFusion(num_classes=2, pretrained=False), CHECKPOINTS_DIR / "early_fusion.pt"),
        "Late Fusion": (LateFusion(num_classes=2, pretrained=False), CHECKPOINTS_DIR / "late_fusion.pt"),
    }

    results = []

    for name, (model, save_path) in models.items():
        print(f"\n{'='*20} Training {name} {'='*20}")
        trained_model, _ = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            epochs=args.epochs,
            lr=args.lr,
            class_weights=weights,
            device=DEVICE,
            save_path=save_path,
            verbose=True,
        )

        test_metrics = evaluate_model(trained_model, test_loader, class_weights=weights, device=DEVICE)
        print(f"[AeroPV-Trust] {name:12s} Test Results -> AUC: {test_metrics['roc_auc']:.3f} | AP: {test_metrics['ap']:.3f} | F1: {test_metrics['f1']:.3f} | BalAcc: {test_metrics['balanced_acc']:.3f}")

        results.append(
            {
                "Model": name,
                "ROC-AUC": round(test_metrics["roc_auc"], 3),
                "AP": round(test_metrics["ap"], 3),
                "F1-score": round(test_metrics["f1"], 3),
                "Balanced Accuracy": round(test_metrics["balanced_acc"], 3),
                "Brier Score": round(test_metrics["brier_score"], 4),
            }
        )

    results_df = pd.DataFrame(results)
    benchmark_path = RESULTS_DIR / "model_benchmarks.csv"
    results_df.to_csv(benchmark_path, index=False)
    print(f"\n[AeroPV-Trust] Saved comprehensive benchmark results to: {benchmark_path}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
