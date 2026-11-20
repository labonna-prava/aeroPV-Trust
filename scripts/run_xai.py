"""
CLI Script: Multi-Level Explainability and Trustworthiness Analysis
Author: Labonna Prava
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.config import CROPS_DIR, CHECKPOINTS_DIR, DEVICE
from src.data.dataset import create_dataloaders
from src.models.late_fusion import LateFusion
from src.explainability.ablation import evaluate_modality_ablation
from src.explainability.prototypes import PrototypeRetriever
from src.explainability.counterfactual import CrossModalCounterfactualExplainer


def main():
    index_csv = CROPS_DIR / "index.csv"
    ckpt_path = CHECKPOINTS_DIR / "late_fusion.pt"

    if not index_csv.exists() or not ckpt_path.exists():
        print("[AeroPV-Trust] Please run scripts/extract_crops.py and scripts/train.py first.")
        return

    train_loader, test_loader, train_df, test_df = create_dataloaders(index_csv=index_csv, batch_size=16)

    model = LateFusion(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    print("\n" + "=" * 25 + " 1. MODALITY ABLATION ANALYSIS " + "=" * 25)
    ablation = evaluate_modality_ablation(model, test_loader, device=DEVICE)
    for k, v in ablation.items():
        print(f"  - {k}: {v:.4f}")

    print("\n" + "=" * 25 + " 2. PROTOTYPE RETRIEVAL INDEX " + "=" * 25)
    retriever = PrototypeRetriever(model, device=DEVICE)
    retriever.build_index(train_loader)
    print(f"  Indexed {len(retriever.metadata_records)} training module representations.")

    # Query first sample from test loader
    sample_rgb, sample_th, sample_y = next(iter(test_loader))
    matches = retriever.query(sample_rgb[0], sample_th[0], top_k=3)
    print(f"  Query Test Sample Label: {sample_y[0].item()}")
    for match in matches:
        print(f"    Rank {match['rank']}: Similarity={match['similarity']:.4f}, Label={match['label']}, AnomalyScore={match['score']:.3f}")

    print("\n" + "=" * 25 + " 3. CROSS-MODAL COUNTERFACTUAL (CCME) " + "=" * 25)
    ccme = CrossModalCounterfactualExplainer(model, device=DEVICE)
    ccme.fit_normal_baseline(train_loader)

    cf_res = ccme.explain(sample_rgb[0], sample_th[0])
    for k, v in cf_res.items():
        print(f"  - {k}: {v}")

    print("\n[AeroPV-Trust] Explainability and trustworthiness suite completed successfully!")


if __name__ == "__main__":
    main()
