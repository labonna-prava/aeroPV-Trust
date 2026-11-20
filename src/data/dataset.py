"""
PyTorch Paired Multimodal Dataset Loader for PV Inspection
Author: Labonna Prava
"""

from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from ..config import IMAGE_SIZE, BATCH_SIZE, RANDOM_SEED, TEST_SIZE


class PairedModuleDataset(Dataset):
    """
    Multimodal Dataset yielding paired (RGB, Thermal, Label) samples.
    RGB is 3-channel (ImageNet normalized), Thermal is 1-channel.
    """

    def __init__(
        self,
        index_df: pd.DataFrame,
        img_size: int = IMAGE_SIZE,
        split: str = "train",
        augment: bool = True,
    ):
        self.df = index_df.reset_index(drop=True)
        self.img_size = img_size
        self.split = split

        aug_transforms = [transforms.RandomHorizontalFlip(p=0.5)] if (split == "train" and augment) else []

        self.rgb_tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                *aug_transforms,
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        self.th_tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                *aug_transforms,
                transforms.ToTensor(),  # Shape: (1, H, W)
                transforms.Normalize((0.5,), (0.25,)),
            ]
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]

        rgb_path = Path(row["rgb_npy"])
        th_path = Path(row["thermal_npy"])

        rgb = np.load(rgb_path.as_posix())  # BGR format
        th = np.load(th_path.as_posix())    # 2D HxW uint8

        # Convert BGR to RGB
        rgb = rgb[:, :, ::-1].copy()
        if th.ndim == 2:
            th = th[:, :, None]

        rgb_tensor = self.rgb_tf(rgb)
        th_tensor = self.th_tf(th)
        label_tensor = torch.tensor(int(row["label"]), dtype=torch.long)

        return rgb_tensor, th_tensor, label_tensor


def create_dataloaders(
    index_csv: Path,
    batch_size: int = BATCH_SIZE,
    img_size: int = IMAGE_SIZE,
    test_size: float = TEST_SIZE,
    seed: int = RANDOM_SEED,
) -> Tuple[DataLoader, DataLoader, pd.DataFrame, pd.DataFrame]:
    """
    Load dataset index, perform stratified train-test splitting,
    and return train/test DataLoader instances.
    """
    df = pd.read_csv(index_csv)

    # Stratified split preserving defect/normal ratio
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["label"]
    )

    train_ds = PairedModuleDataset(train_df, img_size=img_size, split="train", augment=True)
    test_ds = PairedModuleDataset(test_df, img_size=img_size, split="test", augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, drop_last=False
    )

    return train_loader, test_loader, train_df, test_df
