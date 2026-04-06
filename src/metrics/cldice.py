"""Topology-aware clDice for binary vessel masks (hard prediction version)."""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize


def _dilate_bin(mask: np.ndarray, iterations: int = 2) -> np.ndarray:
    m = mask.astype(bool)
    if not m.any():
        return m
    return ndimage.binary_dilation(m, iterations=iterations)


def cldice_score(pred_bin: np.ndarray, gt_bin: np.ndarray, *, dilate_iters: int = 2, eps: float = 1e-6) -> float:
    """
    clDice = 0.5 * (T_prec + T_rec) with
      T_prec = |Skel(P) ∩ Dil(GT)| / |Skel(P)|
      T_rec  = |Skel(GT) ∩ Dil(P)| / |Skel(GT)|
    """
    p = pred_bin.astype(bool)
    g = gt_bin.astype(bool)
    if p.sum() == 0 and g.sum() == 0:
        return 1.0
    if p.sum() == 0 or g.sum() == 0:
        return 0.0

    skel_p = skeletonize(p)
    skel_g = skeletonize(g)
    sp = int(skel_p.sum())
    sg = int(skel_g.sum())
    if sp == 0 and sg == 0:
        return 1.0
    dil_g = _dilate_bin(g, dilate_iters)
    dil_p = _dilate_bin(p, dilate_iters)
    tprec = (skel_p & dil_g).sum() / (sp + eps) if sp else 0.0
    trec = (skel_g & dil_p).sum() / (sg + eps) if sg else 0.0
    return float(0.5 * (tprec + trec))


def mean_cldice_batch(
    pred_logits: np.ndarray,
    gt: np.ndarray,
    threshold: float = 0.5,
    *,
    dilate_iters: int = 2,
) -> float:
    """pred_logits (N,1,H,W) or (N,H,W), gt same shape, values in [0,1]."""
    if pred_logits.ndim == 4:
        pred_logits = pred_logits[:, 0]
    if gt.ndim == 4:
        gt = gt[:, 0]
    prob = 1.0 / (1.0 + np.exp(-pred_logits)) if pred_logits.min() < 0 or pred_logits.max() > 1.0 else pred_logits
    pb = prob >= threshold
    gb = gt >= 0.5
    scores = [cldice_score(pb[i], gb[i], dilate_iters=dilate_iters) for i in range(pb.shape[0])]
    return float(np.mean(scores))
