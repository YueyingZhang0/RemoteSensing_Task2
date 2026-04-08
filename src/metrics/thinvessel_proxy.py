from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize


@dataclass(frozen=True)
class ThinVesselProxyConfig:
    """Dataset-agnostic hard proxy based on thin-vessel skeleton radius."""

    r_th: int = 2
    dilate_iters: int = 1


def thin_vessel_hard_mask(
    gt_vessel: np.ndarray,
    *,
    fov_mask: Optional[np.ndarray] = None,
    cfg: ThinVesselProxyConfig = ThinVesselProxyConfig(),
) -> np.ndarray:
    """
    Build a dataset-agnostic 'hard' mask from GT using the thin-vessel proxy:
      1) skeletonize(GT)
      2) distance_transform_edt(GT)
      3) thin skeleton pixels: radius <= r_th
      4) dilate(thin_skel, 1px) AND GT  (and AND FOV if provided)

    Returns: uint8 mask in {0,1} with same HxW.
    """
    g = (gt_vessel > 0).astype(bool)
    if fov_mask is not None:
        f = (fov_mask > 0).astype(bool)
        g = g & f
    if not g.any():
        return np.zeros_like(gt_vessel, dtype=np.uint8)

    sk = skeletonize(g)
    if not sk.any():
        return np.zeros_like(gt_vessel, dtype=np.uint8)

    # distance to background; on vessel pixels, value approximates local radius in pixels
    dist = ndimage.distance_transform_edt(g)
    thin_skel = sk & (dist <= float(cfg.r_th))
    if not thin_skel.any():
        return np.zeros_like(gt_vessel, dtype=np.uint8)

    if int(cfg.dilate_iters) > 0:
        thin_skel = ndimage.binary_dilation(thin_skel, iterations=int(cfg.dilate_iters))

    hard = thin_skel & g
    return hard.astype(np.uint8)


def thin_proxy_score_from_masks(
    *,
    vessel_mask: np.ndarray,
    hard_mask: np.ndarray,
) -> float:
    """Difficulty score in [0,1]: hard_vessel_pixels / vessel_pixels (0 if vessel empty)."""
    v = (vessel_mask > 0)
    denom = int(v.sum())
    if denom <= 0:
        return 0.0
    h = int(((hard_mask > 0) & v).sum())
    return float(h / denom)


def hard_mask_stats(
    *,
    vessel_mask: np.ndarray,
    hard_mask: np.ndarray,
    fov_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    v = (vessel_mask > 0)
    h = (hard_mask > 0)
    if fov_mask is not None:
        f = (fov_mask > 0)
    else:
        f = np.ones_like(v, dtype=bool)

    fov_pix = int(f.sum())
    vessel_pix = int((v & f).sum())
    hard_pix = int((h & f).sum())
    hard_vessel_pix = int((h & v & f).sum())

    return {
        "hard_pixel_ratio_in_fov": float(hard_pix / fov_pix) if fov_pix else float("nan"),
        "hard_vessel_ratio": float(hard_vessel_pix / vessel_pix) if vessel_pix else float("nan"),
        "counts": {
            "fov_pixels": fov_pix,
            "vessel_pixels_in_fov": vessel_pix,
            "hard_pixels_in_fov": hard_pix,
            "hard_vessel_pixels_in_fov": hard_vessel_pix,
        },
    }


def dice_on_region(
    pred_bin: np.ndarray,
    gt_bin: np.ndarray,
    *,
    region_mask: np.ndarray,
    smooth: float = 1e-5,
) -> float:
    """Dice computed after masking both pred and gt by region_mask."""
    m = (region_mask > 0)
    p = (pred_bin > 0) & m
    g = (gt_bin > 0) & m
    inter = float((p & g).sum())
    return float((2.0 * inter + smooth) / (float(p.sum()) + float(g.sum()) + smooth))

