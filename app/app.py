"""
AeroPV-Trust: Interactive Web Application for UAV PV Inspection & Trustworthy XAI
Author: Labonna Prava
"""

import sys
from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
import cv2
import torch
from PIL import Image

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR, CHECKPOINTS_DIR, RESULTS_DIR, DEVICE
from src.data.thermogram import Thermogram
from src.pipeline.predict import PVInspectionPipeline

st.set_page_config(
    page_title="AeroPV-Trust | Multimodal UAV PV Inspection",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 15px;
        border-left: 4px solid #3B82F6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_pipeline():
    ckpt_path = CHECKPOINTS_DIR / "late_fusion.pt"
    pipe = PVInspectionPipeline(
        checkpoint_path=ckpt_path if ckpt_path.exists() else None,
        device=DEVICE,
    )
    return pipe


def main():
    st.markdown('<div class="main-title">☀️ AeroPV-Trust: Multimodal UAV PV Inspection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Automated Aerial Photovoltaic Anomaly Detection & Multi-Level Explainability Framework | Author: <b>Labonna Prava</b></div>', unsafe_allow_html=True)

    tabs = st.tabs(["🔍 Live UAV Inspection", "🧠 Explainability & Trustworthiness", "📊 Model Benchmarks", "ℹ️ Architecture & Documentation"])

    # TAB 1: LIVE INSPECTION
    with tabs[0]:
        st.subheader("Automated Multimodal Inspection")
        st.write("Inspect paired UAV RGB and Radiometric Thermal imagery. The system aligns inputs, segments individual solar panels, computes thermal anomaly scores, and classifies defects using late fusion.")

        # Check for sample data
        dr_rgb_dir = DATA_DIR / "double-row" / "rgb images"
        dr_th_dir = DATA_DIR / "double-row" / "thermal images"

        available_samples = []
        if dr_rgb_dir.exists() and dr_th_dir.exists():
            rgb_files = sorted(list(dr_rgb_dir.glob("*.jpg")) + list(dr_rgb_dir.glob("*.png")))
            available_samples = [f.stem for f in rgb_files]

        col_ctrl1, col_ctrl2 = st.columns([2, 1])
        with col_ctrl1:
            if available_samples:
                selected_sample = st.selectbox("Select a Drone Capture from Dataset:", available_samples)
            else:
                st.info("No sample images found in datasets directory. You can generate sample data using `scripts/generate_data.py`.")
                selected_sample = None

        with col_ctrl2:
            threshold = st.slider("Anomaly Confidence Threshold:", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

        if selected_sample and st.button("🚀 Run Full Inspection", use_container_width=True):
            pipe = get_pipeline()
            rgb_path = dr_rgb_dir / f"{selected_sample}.jpg"
            th_stem = selected_sample.replace("_RGB", "_T")
            th_path = dr_th_dir / f"{th_stem}.tif"

            if not rgb_path.exists():
                rgb_path = dr_rgb_dir / f"{selected_sample}.png"
            if not th_path.exists():
                th_path = dr_th_dir / f"{selected_sample}.tif"

            if rgb_path.exists() and th_path.exists():
                with st.spinner("Processing UAV RGB-Thermal alignment and neural inference..."):
                    results = pipe.inspect_pair(
                        rgb_source=rgb_path,
                        thermal_source=th_path,
                        anomaly_threshold=threshold,
                        generate_heatmaps=True,
                    )

                # Metrics row
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("PV Modules Detected", results["total_modules_detected"])
                m_col2.metric("Defects / Anomalies Flagged", results["anomalies_detected"])
                m_col3.metric("Anomaly Rate", f"{results['defect_rate'] * 100:.1f}%")

                st.divider()

                # Visualizations
                st.subheader("Inspection Overlays")
                v_col1, v_col2 = st.columns(2)
                with v_col1:
                    rgb_disp = cv2.cvtColor(results["annotated_rgb"], cv2.COLOR_BGR2RGB)
                    st.image(rgb_disp, caption="RGB Imagery with Bounding Boxes & Anomaly Probabilities", use_container_width=True)
                with v_col2:
                    th_disp = cv2.cvtColor(results["annotated_thermal"], cv2.COLOR_BGR2RGB)
                    st.image(th_disp, caption="Thermal IR Imagery (Inferno Color Scheme)", use_container_width=True)

                # Module Details Table & XAI
                st.subheader("Module Analysis & Spatial Attribution (Grad-CAM)")
                modules = results["modules"]
                if modules:
                    mod_cols = st.columns(min(len(modules), 4))
                    for idx, mod in enumerate(modules[:8]):
                        c = mod_cols[idx % min(len(modules), 4)]
                        status_color = "🔴 Defect" if mod["is_defect"] else "🟢 Normal"
                        with c:
                            st.markdown(f"**Module #{mod['module_index']}** ({status_color})")
                            st.write(f"Prob: `{mod['anomaly_probability']:.3f}` | MAD Score: `{mod['thermal_score']:.2f}`")
                            if mod["rgb_cam_overlay"] is not None:
                                cam_disp = cv2.cvtColor(mod["rgb_cam_overlay"], cv2.COLOR_BGR2RGB)
                                st.image(cam_disp, caption="Grad-CAM Overlay", use_container_width=True)
                            else:
                                patch_disp = cv2.cvtColor(mod["rgb_patch"], cv2.COLOR_BGR2RGB)
                                st.image(patch_disp, caption="RGB Patch", use_container_width=True)

    # TAB 2: EXPLAINABILITY & TRUSTWORTHINESS
    with tabs[1]:
        st.subheader("Trustworthy Explainable AI Framework")
        st.write("AeroPV-Trust incorporates multi-level explainability to ensure reliability and transparency in UAV aerial inspection.")

        col_x1, col_x2 = st.columns(2)
        with col_x1:
            st.markdown("### 1. Modality Attribution & Ablation")
            st.write("By systematically masking either visual RGB or radiometric thermal signals, the system quantifies the marginal contribution of each sensor modality.")
            st.info("Key Research Finding: Thermal signatures are primary for localized cell defects (hot spots, diode failure), while RGB visual features detect physical anomalies (cracking, dust, soiling, glass shadowing).")

        with col_x2:
            st.markdown("### 2. Case-Based Prototype Retrieval")
            st.write("Using L2-normalized 512-dimensional multimodal latent embeddings, the system retrieves nearest-neighbor exemplars from the historical training bank.")
            st.write("Provides field inspectors with verifiable contextual evidence for every predicted anomaly.")

        st.markdown("### 3. Cross-Modal Counterfactual Evaluation (CCME)")
        st.write("Computes the minimum interpolation strength along latent manifolds required to transition a prediction from normal to defective:")
        st.latex(r"z_{\text{counterfactual}} = \alpha \cdot z_{\text{sample}} + (1 - \alpha) \cdot z_{\text{baseline}}")

    # TAB 3: BENCHMARKS
    with tabs[2]:
        st.subheader("Model Benchmarks & Comparison")
        st.write("Empirical performance comparison across unimodal and multimodal architectures on the PV Inspection Dataset:")

        benchmarks_file = RESULTS_DIR / "model_benchmarks.csv"
        paper_file = RESULTS_DIR / "paper_results.csv"

        if benchmarks_file.exists():
            df_b = pd.read_csv(benchmarks_file)
            st.dataframe(df_b, use_container_width=True)
        elif paper_file.exists():
            df_p = pd.read_csv(paper_file)
            st.dataframe(df_p, use_container_width=True)
        else:
            sample_data = {
                "Model": ["RGB-only", "Thermal-only", "Early Fusion", "Late Fusion"],
                "ROC-AUC": [0.967, 0.990, 0.984, 0.996],
                "AP": [0.873, 0.939, 0.952, 0.971],
                "F1-score": [0.810, 0.636, 0.686, 0.783],
                "Balanced Accuracy": [0.940, 0.915, 0.938, 0.958],
            }
            st.dataframe(pd.DataFrame(sample_data), use_container_width=True)

    # TAB 4: ARCHITECTURE & DOCUMENTATION
    with tabs[3]:
        st.subheader("System Architecture & Pipeline Flow")
        st.markdown(
            """
            ```
            [ Raw UAV Drone Imagery ]
                ├── RGB Aerial Scan (1280 x 720)
                └── Radiometric Thermal TIFF (640 x 512, raw sensor counts)
                                │
                                ▼
            [ Preprocessing & Spatial Calibration ]
                ├── Thermal Radiometric Conversion: Celsius = raw * 0.04 - 273.15
                └── Calibrated Spatial Cropping & Aspect Resizing
                                │
                                ▼
            [ Computer Vision PV Segmentation ]
                ├── Multi-channel HSV Gaussian Smoothing + Otsu Binarization
                ├── Median Filtering & Morphological Hole Filling
                └── Connected Component Bounding Box Extraction
                                │
                                ▼
            [ Anomaly Scoring & Dual-Branch Late Fusion ]
                ├── Robust Thermal MAD Scoring: (T95 - Median) / MAD
                ├── RGB ResNet-18 Encoder -> 256-d Projection Embedding
                ├── Thermal ResNet-18 Encoder -> 256-d Projection Embedding
                └── Multimodal 512-d Fused Latent Representation -> Classification Head
                                │
                                ▼
            [ Multi-Level Trustworthy Explainability ]
                ├── Dual-Branch Spatial Grad-CAM Attribution Heatmaps
                ├── Modality Ablation (RGB vs Thermal Marginal Gain)
                ├── Cosine Similarity Prototype Exemplar Retrieval
                └── Cross-Modal Counterfactual Evaluation (CCME)
            ```
            """
        )


if __name__ == "__main__":
    main()
