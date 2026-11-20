"""
Explainable AI (XAI) and Trustworthiness modules for AeroPV-Trust.
Author: Labonna Prava
"""

from .gradcam import GradCAM, generate_cam_overlay
from .ablation import evaluate_modality_ablation, compute_modality_contributions
from .prototypes import PrototypeRetriever
from .counterfactual import CrossModalCounterfactualExplainer

__all__ = [
    "GradCAM",
    "generate_cam_overlay",
    "evaluate_modality_ablation",
    "compute_modality_contributions",
    "PrototypeRetriever",
    "CrossModalCounterfactualExplainer",
]
