"""
CLI Script: PV Module Segmentation and Crop Extraction
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
from tqdm import tqdm
import numpy as np

from src.config import DATA_DIR, CROPS_DIR, IR_CROP, RGB_CROP, ANOMALY_PERCENTILE
from src.data.thermogram import load_rgb, load_thermal
from src.data.segmentation import make_aligned_pair, crop_image, extract_module_crops


def build_crops_from_dataset(metadata_csv: Path, out_dir: Path, max_frames: int = 100):
    df = pd.read_csv(metadata_csv)
    dr_dir = metadata_csv.parent

    records = []
    processed = 0

    print(f"[AeroPV-Trust] Extracting module crops from up to {max_frames} frames...")
    for idx, row in tqdm(df.iterrows(), total=min(len(df), max_frames)):
        if processed >= max_frames:
            break

        rgb_path = dr_dir / "rgb images" / row["rgb image name"]
        th_path = dr_dir / "thermal images" / row["thermal image name"]

        if not rgb_path.exists() or not th_path.exists():
            continue

        try:
            rgb = load_rgb(rgb_path)
            th = load_thermal(th_path)
        except Exception:
            continue

        th8 = th.render8()
        rgb_aligned, th8_aligned = make_aligned_pair(rgb, th8)
        thC_crop = crop_image(th.celsius, IR_CROP)

        samples, mask01, labels = extract_module_crops(rgb_aligned, th8_aligned, thC_crop)

        for m_idx, sample in enumerate(samples):
            rgb_save = out_dir / "rgb" / f"frame{idx:05d}_m{m_idx:03d}.npy"
            th_save = out_dir / "thermal" / f"frame{idx:05d}_m{m_idx:03d}.npy"

            rgb_save.parent.mkdir(parents=True, exist_ok=True)
            th_save.parent.mkdir(parents=True, exist_ok=True)

            np.save(rgb_save.as_posix(), sample["rgb_patch"])
            np.save(th_save.as_posix(), sample["th8_patch"])

            records.append(
                {
                    "frame_index": idx,
                    "module_index": m_idx,
                    "rgb_npy": rgb_save.as_posix(),
                    "thermal_npy": th_save.as_posix(),
                    "score": float(sample["score"]),
                    "label": int(sample["label"]),
                    "area": int(sample["area"]),
                }
            )

        processed += 1

    index_df = pd.DataFrame(records)

    # Apply percentile relabeling (top 10% thermal anomaly score) if sufficient samples
    if len(index_df) > 10:
        threshold = index_df["score"].quantile(ANOMALY_PERCENTILE)
        index_df["label"] = (index_df["score"] >= threshold).astype(int)
        print(f"[AeroPV-Trust] Re-calibrated anomaly threshold (top 10%): {threshold:.4f}")

    index_csv = out_dir / "index.csv"
    index_df.to_csv(index_csv, index=False)
    print(f"[AeroPV-Trust] Extracted {len(index_df)} module samples saved to {index_csv}")
    print(f"[AeroPV-Trust] Class balance: {index_df['label'].value_counts().to_dict()}")
    return index_df


def main():
    parser = argparse.ArgumentParser(description="Extract PV module crops from UAV dataset")
    parser.add_argument("--meta", type=str, default=str(DATA_DIR / "double-row" / "metadata.csv"))
    parser.add_argument("--out_dir", type=str, default=str(CROPS_DIR))
    parser.add_argument("--max_frames", type=int, default=100)
    args = parser.parse_args()

    build_crops_from_dataset(Path(args.meta), Path(args.out_dir), args.max_frames)


if __name__ == "__main__":
    main()
