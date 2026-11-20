"""
CLI Script: Full Drone Inspection on Arbitrary Image Pairs
Author: Labonna Prava
"""

import argparse
import sys
from pathlib import Path
import cv2

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR, CHECKPOINTS_DIR, RESULTS_DIR
from src.pipeline.predict import PVInspectionPipeline


def main():
    parser = argparse.ArgumentParser(description="Run end-to-end PV inspection on an RGB+Thermal UAV capture")
    parser.add_argument("--rgb", type=str, default=None, help="Path to UAV RGB image")
    parser.add_argument("--thermal", type=str, default=None, help="Path to radiometric thermal TIFF or 8-bit image")
    parser.add_argument("--threshold", type=float, default=0.5, help="Anomaly decision threshold")
    parser.add_argument("--out_dir", type=str, default=str(RESULTS_DIR / "inspections"), help="Directory to save inspection outputs")
    args = parser.parse_args()

    # Default to first sample if not provided
    if args.rgb is None or args.thermal is None:
        rgb_dir = DATA_DIR / "double-row" / "rgb images"
        th_dir = DATA_DIR / "double-row" / "thermal images"
        samples = list(rgb_dir.glob("*.jpg"))
        if not samples:
            print("No samples found. Please provide --rgb and --thermal.")
            return
        args.rgb = str(samples[0])
        th_stem = samples[0].stem.replace("_RGB", "_T")
        args.thermal = str(th_dir / f"{th_stem}.tif")

    print(f"[AeroPV-Trust] Running inspection on:")
    print(f"  RGB: {args.rgb}")
    print(f"  Thermal: {args.thermal}")

    ckpt_path = CHECKPOINTS_DIR / "late_fusion.pt"
    pipe = PVInspectionPipeline(checkpoint_path=ckpt_path if ckpt_path.exists() else None)

    res = pipe.inspect_pair(
        rgb_source=Path(args.rgb),
        thermal_source=Path(args.thermal),
        anomaly_threshold=args.threshold,
        generate_heatmaps=True,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    annotated_rgb_path = out_dir / "annotated_rgb.jpg"
    annotated_th_path = out_dir / "annotated_thermal.jpg"
    cv2.imwrite(str(annotated_rgb_path), res["annotated_rgb"])
    cv2.imwrite(str(annotated_th_path), res["annotated_thermal"])

    print(f"\n[AeroPV-Trust] Inspection Summary:")
    print(f"  Total Modules Detected : {res['total_modules_detected']}")
    print(f"  Anomalies Flagged      : {res['anomalies_detected']}")
    print(f"  Anomaly Rate           : {res['defect_rate']*100:.1f}%")
    print(f"  Saved Overlays to      : {out_dir}\n")

    for m in res["modules"][:6]:
        status = "DEFECT" if m["is_defect"] else "NORMAL"
        print(f"  Module {m['module_index']:02d} | BBox: {m['bbox']} | Prob: {m['anomaly_probability']:.4f} | Status: {status}")


if __name__ == "__main__":
    main()
