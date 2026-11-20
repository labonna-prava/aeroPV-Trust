"""
CLI Script: Generate Sample UAV Dataset for Immediate Execution
Author: Labonna Prava
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR
from src.data.generator import generate_synthetic_uav_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic multimodal UAV dataset for PV inspection")
    parser.add_argument("--samples", type=int, default=40, help="Number of paired UAV frames to generate")
    parser.add_argument("--out_dir", type=str, default=str(DATA_DIR), help="Output dataset directory")
    args = parser.parse_args()

    print(f"[AeroPV-Trust] Generating {args.samples} synthetic UAV frames in: {args.out_dir}")
    df = generate_synthetic_uav_dataset(data_dir=Path(args.out_dir), num_samples=args.samples)
    print(f"[AeroPV-Trust] Successfully generated {len(df)} paired records with RGB, 16-bit thermal TIFF, and metadata.")


if __name__ == "__main__":
    main()
