from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.metrics.thinvessel_proxy import (
    ThinVesselProxyConfig,
    thin_proxy_score_from_masks,
    thin_vessel_hard_mask,
)


@dataclass(frozen=True)
class ExternalCropConfig:
    image_size: int = 128
    crops_per_image: int = 64  # dataset length multiplier
    thin_cfg: ThinVesselProxyConfig = ThinVesselProxyConfig(r_th=2, dilate_iters=1)


def _read_rgb01(p: Path) -> np.ndarray:
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0


def _read_mask01(p: Path) -> np.ndarray:
    arr = np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0
    return (arr > 0.5).astype(np.uint8)


class ExternalThinVesselCropDataset(Dataset):
    """Random crops from external whole images with thin-vessel proxy score as 'sci_res2_norm'."""

    def __init__(
        self,
        *,
        cases: List[Dict[str, Path]],
        cfg: ExternalCropConfig,
        seed: int,
        augment: bool,
    ) -> None:
        self.cases = cases
        self.cfg = cfg
        self.seed = int(seed)
        self.augment = bool(augment)

        # Preload GT-derived masks used for proxy score (hard mask within GT; optional FOV).
        self._cache: List[Dict[str, Any]] = []
        for c in self.cases:
            img = Path(c["image"])
            gt = Path(c["gt"])
            fov = Path(c["fov"]) if "fov" in c and c["fov"] is not None else None
            gt_m = _read_mask01(gt)
            fov_m = _read_mask01(fov) if fov is not None else None
            hard = thin_vessel_hard_mask(gt_m, fov_mask=fov_m, cfg=self.cfg.thin_cfg)
            self._cache.append(
                {
                    "case_id": str(c["case_id"]),
                    "image": img,
                    "gt": gt,
                    "fov": fov,
                    "gt_mask": gt_m,
                    "fov_mask": fov_m,
                    "hard_mask": hard,
                }
            )

    def __len__(self) -> int:
        return len(self._cache) * int(self.cfg.crops_per_image)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Deterministic RNG per index for reproducibility across workers (num_workers=0 by default).
        rng = np.random.default_rng(self.seed + int(idx))
        rec = self._cache[int(idx) % len(self._cache)]

        rgb = _read_rgb01(rec["image"])
        gt = rec["gt_mask"]
        fov = rec["fov_mask"]
        hard = rec["hard_mask"]

        h, w, _ = rgb.shape
        isz = int(self.cfg.image_size)
        if h < isz or w < isz:
            pad_h = max(0, isz - h)
            pad_w = max(0, isz - w)
            rgb = np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            gt = np.pad(gt, ((0, pad_h), (0, pad_w)), mode="constant")
            hard = np.pad(hard, ((0, pad_h), (0, pad_w)), mode="constant")
            if fov is not None:
                fov = np.pad(fov, ((0, pad_h), (0, pad_w)), mode="constant")
            h, w, _ = rgb.shape

        y = int(rng.integers(0, h - isz + 1))
        x = int(rng.integers(0, w - isz + 1))

        img_c = rgb[y : y + isz, x : x + isz, :]
        gt_c = gt[y : y + isz, x : x + isz]
        hard_c = hard[y : y + isz, x : x + isz]
        if fov is not None:
            fov_c = fov[y : y + isz, x : x + isz]
            gt_c = (gt_c > 0) & (fov_c > 0)
            hard_c = (hard_c > 0) & (fov_c > 0)
        else:
            gt_c = gt_c > 0
            hard_c = hard_c > 0

        if self.augment:
            if rng.random() > 0.5:
                img_c = np.fliplr(img_c).copy()
                gt_c = np.fliplr(gt_c).copy()
                hard_c = np.fliplr(hard_c).copy()
            if rng.random() > 0.5:
                img_c = np.flipud(img_c).copy()
                gt_c = np.flipud(gt_c).copy()
                hard_c = np.flipud(hard_c).copy()

        image = torch.from_numpy(img_c.astype(np.float32)).permute(2, 0, 1)
        mask = torch.from_numpy(gt_c.astype(np.float32)).unsqueeze(0)
        score = thin_proxy_score_from_masks(vessel_mask=gt_c.astype(np.uint8), hard_mask=hard_c.astype(np.uint8))
        sce = torch.tensor([float(score)], dtype=torch.float32)

        return {
            "image": image,
            "mask": mask,
            "sce": sce,
            "sci_res2_norm": float(score),
            "fov_ratio": 1.0,
            "patch_name": f"{rec['case_id']}@y{y}_x{x}",
            "sample_id": str(rec["case_id"]),
            "is_hard": False,  # not used for external fine-tune
        }

