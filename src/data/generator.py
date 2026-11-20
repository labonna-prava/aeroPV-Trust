"""
Synthetic Multimodal UAV Data Generator
Author: Labonna Prava

Generates realistic UAV RGB and radiometric Thermal frames with solar panel rows,
sub-module structures, dust/soiling, and localized hot-spot anomalies.
Allows the full AeroPV-Trust pipeline (crops, training, evaluation, XAI) to be run
immediately on any system, even before downloading the large external Kaggle dataset.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import tifffile as tif
from ..config import DATA_DIR, IR_CROP, RGB_CROP


def generate_synthetic_uav_dataset(
    data_dir: Path = DATA_DIR,
    num_samples: int = 40,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Create a complete synthetic single-row and double-row UAV PV dataset
    matching the exact schema, folder structure, and radiometric values
    of the Kaggle Photovoltaic System O&M Inspection dataset.
    """
    np.random.seed(seed)
    data_dir = Path(data_dir)

    dr_dir = data_dir / "double-row"
    dr_rgb_dir = dr_dir / "rgb images"
    dr_th_dir = dr_dir / "thermal images"
    sr_dir = data_dir / "single-row"
    sr_th_dir = sr_dir / "thermal images"

    for d in [dr_rgb_dir, dr_th_dir, sr_th_dir]:
        d.mkdir(parents=True, exist_ok=True)

    records = []
    rgb_w, rgb_h = 1280, 720
    ir_w, ir_h = 640, 512

    for i in range(num_samples):
        # Base ambient temperatures in Celsius: ~30°C to 45°C
        base_temp_c = np.random.uniform(32.0, 38.0)
        
        # 1. Generate RGB UAV Frame (Double Row layout)
        rgb_img = np.full((rgb_h, rgb_w, 3), (85, 105, 75), dtype=np.uint8) # Grass background
        # Add terrain noise
        noise = np.random.normal(0, 8, (rgb_h, rgb_w, 3)).astype(np.int16)
        rgb_img = np.clip(rgb_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # 2. Generate Thermal Frame (radiometric 16-bit raw counts)
        # raw = (Celsius + 273.15) / 0.04
        th_celsius = np.full((ir_h, ir_w), base_temp_c, dtype=np.float32)
        th_celsius += np.random.normal(0, 0.4, (ir_h, ir_w)).astype(np.float32)

        # Draw PV panel modules
        # Panel grid specifications
        n_rows = 2
        n_cols = 6
        panel_w_rgb, panel_h_rgb = 120, 220
        start_x_rgb, start_y_rgb = 200, 100

        has_defect = (np.random.rand() > 0.4)
        defect_col = np.random.randint(0, n_cols)
        defect_row = np.random.randint(0, n_rows)

        # Draw panels on RGB
        for r in range(n_rows):
            for c in range(n_cols):
                px1 = start_x_rgb + c * (panel_w_rgb + 20)
                py1 = start_y_rgb + r * (panel_h_rgb + 40)
                px2 = px1 + panel_w_rgb
                py2 = py1 + panel_h_rgb

                # Panel frame (aluminum silver)
                cv2.rectangle(rgb_img, (px1, py1), (px2, py2), (180, 180, 180), 2)
                # Dark blue monocrystalline PV silicon surface
                cv2.rectangle(rgb_img, (px1 + 2, py1 + 2), (px2 - 2, py2 - 2), (120, 50, 20), -1)

                # Grid cell busbars
                for b in range(1, 4):
                    bx = px1 + b * (panel_w_rgb // 4)
                    cv2.line(rgb_img, (bx, py1 + 2), (bx, py2 - 2), (160, 160, 160), 1)

                # Add visual anomaly (dust, discoloration, cracked cell)
                if has_defect and r == defect_row and c == defect_col:
                    cv2.circle(rgb_img, (px1 + panel_w_rgb // 2, py1 + panel_h_rgb // 2), 22, (50, 70, 90), -1)

        # Draw corresponding panels on Thermal IR
        # Thermal mapped bounding regions
        t_crop_w = ir_w
        t_crop_h = ir_h - 75
        for r in range(n_rows):
            for c in range(n_cols):
                # Map coordinates proportionally to thermal frame
                tx1 = int(70 + c * 85)
                ty1 = int(120 + r * 150)
                tx2 = tx1 + 75
                ty2 = ty1 + 130

                # Normal operating panel: ~42°C
                panel_temp = base_temp_c + np.random.uniform(8.0, 11.0)
                th_celsius[ty1:ty2, tx1:tx2] = panel_temp

                # If this module has an anomaly (hot spot, bypass diode failure): ~60°C to 80°C!
                if has_defect and r == defect_row and c == defect_col:
                    hot_spot_temp = base_temp_c + np.random.uniform(25.0, 45.0)
                    cx, cy = (tx1 + tx2) // 2, (ty1 + ty2) // 2
                    # Create Gaussian hot spot
                    y_indices, x_indices = np.ogrid[:ir_h, :ir_w]
                    dist_sq = (x_indices - cx) ** 2 + (y_indices - cy) ** 2
                    mask_hot = dist_sq < (18 ** 2)
                    th_celsius[mask_hot] = hot_spot_temp

        # Convert Celsius back to 16-bit raw counts for radiometric TIFF
        # raw = (Celsius + 273.15) / 0.04
        raw_counts = ((th_celsius + 273.15) / 0.04).astype(np.uint16)

        rgb_filename = f"DJI_{i:04d}_RGB.jpg"
        th_filename = f"DJI_{i:04d}_T.tif"

        rgb_path = dr_rgb_dir / rgb_filename
        th_path = dr_th_dir / th_filename

        cv2.imwrite(rgb_path.as_posix(), rgb_img)
        tif.imwrite(th_path.as_posix(), raw_counts)

        records.append(
            {
                "thermal image name": th_filename,
                "rgb image name": rgb_filename,
                "timestamp": f"2026-05-15 11:{i%60:02d}:00",
                "ambient_temp": round(base_temp_c, 2),
                "has_defect": int(has_defect),
            }
        )

    # Save double-row metadata
    dr_df = pd.DataFrame(records)
    dr_df.to_csv(dr_dir / "metadata.csv", index=False)

    # Also generate single-row metadata for completeness
    sr_df = dr_df[["thermal image name", "timestamp", "ambient_temp"]].copy()
    sr_df.to_csv(sr_dir / "metadata.csv", index=False)

    return dr_df
