from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

_META_COLS = [
    "vessel_area",
    "fov_ratio",
    "junction_count",
    "component_count",
    "endpoint_eff",
]


class PatchRegressionDataset(Dataset):
    """Returns dict per sample; used by Task 2 SCE probe."""

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str | Path,
        target_col: str,
        image_size: int = 128,
        augment: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.target_col = target_col
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        name = str(row["patch_name"])
        path = self.image_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing patch image: {path}")

        img = Image.open(path).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0

        if self.augment:
            if np.random.rand() > 0.5:
                arr = np.fliplr(arr).copy()
            if np.random.rand() > 0.5:
                arr = np.flipud(arr).copy()

        image = torch.from_numpy(arr).permute(2, 0, 1)
        target = torch.tensor([float(row[self.target_col])], dtype=torch.float32)

        meta: Dict[str, Any] = {"patch_name": name}
        for c in _META_COLS:
            if c in row.index:
                meta[c] = row[c]

        sample_id = str(row.get("sample_id", ""))

        return {
            "image": image,
            "target": target,
            "patch_name": name,
            "sample_id": sample_id,
            "meta": meta,
        }
