"""
Nearest-Neighbor Multimodal Prototype Retrieval
Author: Labonna Prava
"""

from typing import List, Dict, Optional
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ..config import DEVICE


class PrototypeRetriever:
    """
    Retrieves most similar training prototypes in learned multimodal embedding space.
    Provides case-based interpretable explanations for any query PV module.
    """

    def __init__(self, late_model: torch.nn.Module, device: str = DEVICE):
        self.model = late_model.eval().to(device)
        self.device = device
        self.embedding_bank: Optional[torch.Tensor] = None
        self.metadata_records: List[Dict] = []

    @torch.no_grad()
    def build_index(self, dataloader: DataLoader):
        """Extract and normalize 512-d embeddings for all training modules."""
        self.model.eval()
        embeddings = []
        records = []

        ds = dataloader.dataset
        df = ds.df

        idx = 0
        for rgb, th, y in dataloader:
            rgb = rgb.to(self.device)
            th = th.to(self.device)
            _, _, z = self.model.get_embeddings(rgb, th)

            # L2 normalization for cosine similarity
            z_norm = F.normalize(z, p=2, dim=1)
            embeddings.append(z_norm.cpu())

            batch_size = len(y)
            for b in range(batch_size):
                row = df.iloc[idx + b]
                records.append(
                    {
                        "rgb_npy": row.get("rgb_npy"),
                        "thermal_npy": row.get("thermal_npy"),
                        "label": int(y[b].item()),
                        "score": float(row.get("score", 0.0)),
                    }
                )
            idx += batch_size

        self.embedding_bank = torch.cat(embeddings, dim=0)  # [N, 512]
        self.metadata_records = records

    @torch.no_grad()
    def query(
        self,
        rgb_tensor: torch.Tensor,
        th_tensor: torch.Tensor,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Query top-k nearest neighbor prototypes based on cosine similarity.
        """
        if self.embedding_bank is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        rgb = rgb_tensor.unsqueeze(0).to(self.device)
        th = th_tensor.unsqueeze(0).to(self.device)

        _, _, z_query = self.model.get_embeddings(rgb, th)
        z_query_norm = F.normalize(z_query, p=2, dim=1).cpu()

        # Cosine similarity matrix: [1, N]
        sims = torch.mm(z_query_norm, self.embedding_bank.t()).squeeze(0)
        top_indices = torch.topk(sims, k=min(top_k, len(self.metadata_records))).indices.tolist()

        results = []
        for rank, match_idx in enumerate(top_indices):
            rec = dict(self.metadata_records[match_idx])
            rec["similarity"] = float(sims[match_idx].item())
            rec["rank"] = rank + 1
            results.append(rec)

        return results
