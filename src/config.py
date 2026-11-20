"""
Global Configuration and Hyperparameters for AeroPV-Trust
Author: Labonna Prava
"""

import os
from pathlib import Path
import torch

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("PV_DATA_ROOT", BASE_DIR / "datasets")).expanduser().resolve()
CROPS_DIR = Path(os.environ.get("PV_CROPS_ROOT", BASE_DIR / "pv_module_crops")).expanduser().resolve()
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"

# Ensure directories exist
CROPS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset subpaths (matches Kaggle Photovoltaic System O&M Inspection structure)
SR_META = DATA_DIR / "single-row" / "metadata.csv"
DR_META = DATA_DIR / "double-row" / "metadata.csv"
SR_THERM_DIR = DATA_DIR / "single-row" / "thermal images"
DR_THERM_DIR = DATA_DIR / "double-row" / "thermal images"
DR_RGB_DIR = DATA_DIR / "double-row" / "rgb images"

# Spatial crop parameters for raw UAV frames
IR_CROP = dict(left=0, top=75, right=640, bottom=490)
RGB_CROP = dict(left=120, top=0, right=1150, bottom=720)

# Training Hyperparameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
IMAGE_SIZE = 224
RANDOM_SEED = 42
TEST_SIZE = 0.27
ANOMALY_PERCENTILE = 0.90  # Top 10% thermal anomaly score threshold
MAD_EPS = 1e-6
