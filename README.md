# ☀️ AeroPV-Trust: Multimodal UAV Photovoltaic Anomaly Inspection & Trustworthy XAI Framework

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/Author-Labonna%20Prava-6f42c1.svg)](https://github.com/labonna-prava)

An end-to-end aerial computer vision and deep learning system developed by **Labonna Prava** for automated inspection, defect localization, and trustworthy multi-level explainability of photovoltaic (PV) solar farms using multimodal Unmanned Aerial Vehicle (UAV) RGB and radiometric thermal infrared imagery.

---

## 📌 Executive Overview

Utility-scale solar farms require continuous operations and maintenance (O&M) monitoring to maintain peak efficiency. Traditional manual thermography is labor-intensive and prone to human error, while standard single-sensor automated approaches suffer from severe limitations:
- **Visual RGB imagery alone** can identify physical anomalies (soiling, shading, physical glass cracking) but fails to detect critical internal electrical failures, cell micro-cracks, and bypass diode short circuits.
- **Thermal infrared (IR) imagery alone** highlights thermal hot spots but struggles to differentiate genuine electrical defects from harmless surface shadows or bird droppings.

**AeroPV-Trust** bridges this gap through a unified deep learning and computer vision framework that:
1. Performs automated **spatial alignment** and **radiometric calibration** of drone-acquired thermal and RGB frames.
2. Extracts individual PV solar modules via **multi-channel HSV Gaussian Otsu thresholding** and morphological connected components.
3. Scores each module with a robust **Median Absolute Deviation (MAD)** thermal anomaly metric.
4. Classifies defects using a **dual-branch late fusion architecture** alongside comparative unimodal (RGB-only, Thermal-only) and early fusion baselines.
5. Implements a **four-tier Trustworthy Explainable AI (XAI)** suite:
   - **Multi-Level Grad-CAM:** Dual-branch spatial activation mapping for both visual and thermal channels.
   - **Modality Ablation Analysis:** Quantifying marginal feature importance by systematically masking sensor modalities.
   - **Multimodal Prototype Retrieval:** Case-based cosine similarity retrieval in the 512-dimensional fused latent space.
   - **Cross-Modal Counterfactual Evaluation (CCME):** Estimating latent boundary transitions between normal and anomalous states.
6. Delivers an interactive **Streamlit Web Application** for real-time field inspection and interactive model interpretability.

---

## 🏛️ System Architecture

```text
               ┌─────────────────────────────────────────────────────────┐
               │              Raw UAV Drone Aerial Capture               │
               │   • High-Resolution RGB Imagery (1280 × 720)            │
               │   • Radiometric Thermal 16-bit TIFF (640 × 512)         │
               └────────────────────────────┬────────────────────────────┘
                                            │
                                            ▼
               ┌─────────────────────────────────────────────────────────┐
               │          Preprocessing & Radiometric Calibration        │
               │   • Celsius Calibration: T(°C) = (raw * 0.04) - 273.15   │
               │   • 8-bit Min-Max Normalization & Inferno Colormapping  │
               │   • Spatial Aspect Cropping & Coordinate Alignment      │
               └────────────────────────────┬────────────────────────────┘
                                            │
                                            ▼
               ┌─────────────────────────────────────────────────────────┐
               │             Computer Vision PV Segmentation             │
               │   • Multi-Channel HSV Otsu Binarization                 │
               │   • Median Blurring & Morphological Hole-Filling        │
               │   • Connected-Component Bounding Box Extraction         │
               └────────────────────────────┬────────────────────────────┘
                                            │
                                            ▼
               ┌─────────────────────────────────────────────────────────┐
               │           Thermal Scoring & Paired Module Crops         │
               │   • Anomaly Score = (T95(module) - Median) / MAD        │
               │   • Paired 224 × 224 RGB and Thermal Module Patches     │
               └────────────────────────────┬────────────────────────────┘
                                            │
                                            ▼
               ┌─────────────────────────────────────────────────────────┐
               │          Dual-Branch Late Fusion Deep Learning          │
               │   • RGB ResNet-18 Encoder     ──► 256-d Projection      │
               │   • Thermal ResNet-18 Encoder ──► 256-d Projection      │
               │   • Concatenated Latent Vector ──► 512-d Fused Vector   │
               │   • Shared Classification Head ──► Anomaly Probability  │
               └────────────────────────────┬────────────────────────────┘
                                            │
                                            ▼
               ┌─────────────────────────────────────────────────────────┐
               │               Trustworthy Multi-Level XAI               │
               │   ├── 1. Dual-Branch Grad-CAM Spatial Attribution       │
               │   ├── 2. Modality Ablation (RGB vs Thermal Drops)       │
               │   ├── 3. Case-Based Multimodal Prototype Retrieval      │
               │   └── 4. Cross-Modal Counterfactual Evaluation (CCME)   │
               └─────────────────────────────────────────────────────────┘
```

---

## 🔬 Benchmark Results

The four comparative architectures were systematically trained and evaluated under class-imbalanced settings using class-weighted cross-entropy loss:

| Model Architecture | Input Modality | ROC-AUC | Average Precision (AP) | F1-Score | Balanced Accuracy | Brier Reliability |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **RGB-only** | 3-Channel RGB | 0.967 | 0.873 | **0.810** | 0.940 | 0.0505 |
| **Thermal-only** | 1-Channel Radiometric Thermal | 0.990 | 0.939 | 0.636 | 0.915 | 0.0578 |
| **Early Fusion** | 4-Channel Concatenated (RGB + Thermal) | 0.984 | 0.952 | 0.686 | 0.938 | 0.2495 |
| **Late Fusion (Ours)** | **Dual-Branch Modality Encoders** | **0.996** | **0.971** | 0.783 | **0.958** | **0.0401** |

> **Key Findings:**
> - Late Fusion achieves the highest discrimination (**0.996 ROC-AUC** and **0.971 AP**) by allowing both visual and thermal representations to develop specialized features before merging.
> - Late Fusion demonstrates superior probability calibration with the lowest **Brier Score (0.0401)**, confirming high confidence reliability for real-world field deployment.

---

## 🚀 Quickstart & Installation

### 1. Clone Repository & Setup Environment
```bash
git clone https://github.com/labonna-prava/AeroPV-Trust.git
cd AeroPV-Trust

# Create and activate Python virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Sample Drone Data (Immediate Offline Test)
AeroPV-Trust includes a synthetic multimodal drone generator that creates realistic single- and double-row captures matching the Kaggle schema:
```bash
python scripts/generate_data.py --samples 35
```

### 3. Segment PV Modules & Extract Crops
```bash
python scripts/extract_crops.py --max_frames 35
```

### 4. Train All 4 Model Architectures
```bash
python scripts/train.py --epochs 6 --batch_size 16
```
Trained checkpoints will be saved in `checkpoints/` (`rgb_only.pt`, `thermal_only.pt`, `early_fusion.pt`, `late_fusion.pt`).

### 5. Evaluate Models & Verify Metrics
```bash
python scripts/evaluate.py
```

### 6. Run Explainability & Trustworthiness Suite
```bash
python scripts/run_xai.py
```

### 7. Run Full Drone Inspection on Any Image Pair
```bash
python scripts/run_inspection.py
```

### 8. Launch Interactive Streamlit Web Dashboard
```bash
streamlit run app/app.py
```

---

## 🌐 Kaggle Dataset Integration

This project is fully compatible with the open-access **Photovoltaic System O&M Inspection Dataset** by Marcos Gabriel on Kaggle:
- **Kaggle Link:** [Photovoltaic System O&M Inspection](https://www.kaggle.com/code/marcosgabriel/dataset-intro-photovoltaic-system-o-m-inspection)

To train on the full real-world dataset:
1. Download the dataset zip from Kaggle.
2. Unzip into the `./datasets/` folder with this structure:
   ```text
   datasets/
   ├── single-row/
   │   ├── metadata.csv
   │   └── thermal images/
   │       └── DJI_xxxx_T.tif
   └── double-row/
       ├── metadata.csv
       ├── rgb images/
       │   └── DJI_xxxx_RGB.jpg
       └── thermal images/
           └── DJI_xxxx_T.tif
   ```
3. Run `python scripts/extract_crops.py` to extract all module crops from the full dataset.
4. Run `python scripts/train.py --epochs 8` to train models on the full benchmark.

---

## 📂 Project Directory Structure

```text
AeroPV-Trust/
├── app/
│   └── app.py                     # Interactive Streamlit Web Dashboard
├── checkpoints/                   # Saved trained PyTorch model weights (.pt)
│   ├── rgb_only.pt
│   ├── thermal_only.pt
│   ├── early_fusion.pt
│   └── late_fusion.pt
├── notebooks/
│   ├── aeropv_trust_research_workflow.ipynb   # Interactive research walkthrough
│   └── trustworthy_uav_pv_inspection_research.ipynb
├── results/                       # Empirical benchmarks, evaluations, and inspection outputs
│   ├── model_benchmarks.csv
│   ├── evaluation_summary.csv
│   └── paper_results.csv
├── scripts/                       # CLI commands for reproducible execution
│   ├── generate_data.py           # Sample UAV dataset generator
│   ├── extract_crops.py           # PV segmentation and patch extraction
│   ├── train.py                   # Automated training across 4 architectures
│   ├── evaluate.py                # Model validation and metric comparison
│   ├── run_xai.py                 # Grad-CAM, Ablation, Prototypes, and CCME
│   └── run_inspection.py          # End-to-end drone inspection pipeline
├── src/                           # Production-grade Python package
│   ├── __init__.py
│   ├── config.py                  # Global hyperparameters and directory paths
│   ├── data/
│   │   ├── thermogram.py          # Radiometric calibration & Celsius conversion
│   │   ├── segmentation.py        # HSV Otsu binarization & MAD anomaly scoring
│   │   ├── dataset.py             # PyTorch PairedModuleDataset & stratified split
│   │   └── generator.py           # Realistic synthetic drone imagery generator
│   ├── models/
│   │   ├── backbones.py           # Modified ResNet-18 feature extractors
│   │   ├── rgb_model.py           # Unimodal RGB-only classifier
│   │   ├── thermal_model.py       # Unimodal Thermal-only classifier
│   │   ├── early_fusion.py        # 4-channel Early Fusion classifier
│   │   └── late_fusion.py         # Dual-branch Late Fusion classifier
│   ├── training/
│   │   ├── trainer.py             # Training loop with class-weighted cross-entropy
│   │   └── metrics.py             # ROC-AUC, AP, F1, Balanced Accuracy, Brier score
│   ├── explainability/
│   │   ├── gradcam.py             # Multi-level Grad-CAM context manager
│   │   ├── ablation.py            # Modality ablation and dominance analysis
│   │   ├── prototypes.py          # Cosine similarity nearest-neighbor retrieval
│   │   └── counterfactual.py      # Cross-Modal Counterfactual Evaluation (CCME)
│   └── pipeline/
│       └── predict.py             # High-level UAV inspection pipeline
├── .gitignore
├── LICENSE                        # MIT License
├── README.md                      # Comprehensive project documentation
└── requirements.txt               # Python package dependencies
```

---

## 👩‍💻 Author & Maintainer

Developed by **Labonna Prava**  
- **GitHub:** [@labonna-prava](https://github.com/labonna-prava)  
- **Project Name:** AeroPV-Trust  
- **Domain:** Computer Vision, Aerial Thermography, Renewable Energy AI & Trustworthy Machine Learning

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
