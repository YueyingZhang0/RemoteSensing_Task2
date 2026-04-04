from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class SegSCEPatchDataset(Dataset):
    """RGB patch + vessel mask (binary) + sci_res2_norm scalar."""

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str | Path,
        mask_dir: str | Path,
        target_col: str = "sci_res2_norm",
        image_size: int = 128,
        augment: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.target_col = target_col
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        name = str(row["patch_name"])
        ip = self.image_dir / name
        mp = self.mask_dir / name
        if not ip.is_file():
            raise FileNotFoundError(ip)
        if not mp.is_file():
            raise FileNotFoundError(mp)

        img = np.asarray(Image.open(ip).convert("RGB"), dtype=np.float32) / 255.0
        msk = np.asarray(Image.open(mp).convert("L"), dtype=np.float32) / 255.0
        msk = (msk > 0.5).astype(np.float32)

        if self.augment:
            if np.random.rand() > 0.5:
                img = np.fliplr(img).copy()
                msk = np.fliplr(msk).copy()
            if np.random.rand() > 0.5:
                img = np.flipud(img).copy()
                msk = np.flipud(msk).copy()

        image = torch.from_numpy(img).permute(2, 0, 1)
        mask = torch.from_numpy(msk).unsqueeze(0)
        sce = torch.tensor([float(row[self.target_col])], dtype=torch.float32)

        out: Dict[str, Any] = {
            "image": image,
            "mask": mask,
            "sce": sce,
            "patch_name": name,
            "sample_id": str(row.get("sample_id", "")),
        }
        if "is_hard" in row.index:
            out["is_hard"] = bool(row["is_hard"])
        return out
