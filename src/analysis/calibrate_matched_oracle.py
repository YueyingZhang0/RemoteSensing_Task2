#!/usr/bin/env python3
"""Calibrate oracle tau / alpha / max_weight to match frozen-probe trigger rate and weight stats (train split)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.datasets.patch_regression_dataset import PatchRegressionDataset
from src.training.splits import load_split_info
from src.training.task3_engine import load_frozen_sce_probe, oracle_sample_weights
from src.utils.io import load_yaml, save_json


def _train_df(root: Path, cfg: dict) -> pd.DataFrame:
    meta = Path(cfg["metadata_csv"])
    if not meta.is_absolute():
        meta = root / meta
    split_json = cfg.get("split_json")
    if not split_json:
        raise ValueError("split_json required")
    sp = Path(split_json)
    if not sp.is_absolute():
        sp = root / sp
    info = load_split_info(sp)
    train_ids = set(str(x) for x in info["train_sample_ids"])
    df = pd.read_csv(meta)
    df = df[df["sample_id"].astype(str).isin(train_ids)].reset_index(drop=True)
    return df


def _simulate_weight_stats(
    scores: np.ndarray,
    batch_size: int,
    tau: float,
    alpha: float,
    gamma: float,
    min_w: float,
    max_w: float,
    normalize: bool,
) -> Tuple[float, float, float]:
    """Return (mean, std, max) of per-sample weights across all batches (same as training aggregation)."""
    s = scores.astype(np.float64)
    n = len(s)
    all_w: list[float] = []
    for start in range(0, n, batch_size):
        chunk = s[start : start + batch_size]
        t = torch.tensor(chunk, dtype=torch.float32)
        w = oracle_sample_weights(t, tau, alpha, gamma, min_w, max_w, normalize)
        all_w.extend(w.detach().numpy().ravel().tolist())
    arr = np.asarray(all_w, dtype=np.float64)
    return float(arr.mean()), float(arr.std()), float(arr.max())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/task3_v3_1_predicted_frozen.yaml")
    p.add_argument("--probe_ckpt", type=str, default="outputs/task3_difficulty_probe/best_sce_branch.pt")
    p.add_argument("--tau_probe", type=float, default=0.6, help="Frozen probe difficulty tau for trigger match")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--out_json", type=str, default="outputs/task3_v3_1_predicted_frozen_same_split/matched_oracle_calibration.json")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[2]
    cfg = load_yaml(root / args.config)
    df_tr = _train_df(root, cfg)
    target_col = cfg.get("target_col", "sci_res2_norm")
    img_dir = Path(cfg["patch_image_dir"])
    if not img_dir.is_absolute():
        img_dir = root / img_dir
    image_size = int(cfg.get("image_size", 128))

    gt = df_tr[target_col].astype(np.float64).values

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = Path(args.probe_ckpt)
    if not ck.is_absolute():
        ck = root / ck
    probe = load_frozen_sce_probe(ck, dev, in_channels=3, base_channels=32, mlp_hidden=64, dropout=0.0)
    probe.eval()

    ds = PatchRegressionDataset(df_tr, img_dir, target_col, image_size=image_size, augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    preds: list[float] = []
    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(dev)
            out = probe(img)["sci_pred"].squeeze(1).clamp(0.0, 1.0)
            preds.extend(out.cpu().numpy().ravel().tolist())
    pred_arr = np.asarray(preds, dtype=np.float64)
    target_frac = float((pred_arr > float(args.tau_probe)).mean())
    # tau such that fraction of GT scores above tau equals target_frac
    tau_trigger = float(np.quantile(gt, 1.0 - target_frac))

    gamma = 2.0
    min_w = 1.0
    normalize = True
    dw = cfg.get("difficulty_weighting") or {}
    fp_tau = float(dw.get("tau", 0.6))
    fp_alpha = float(dw.get("alpha", 0.5))
    fp_max = float(dw.get("max_weight", 1.5))
    # Target stats = actual frozen-probe weight pool on train (pred -> oracle_sample_weights)
    target_mean, target_std, target_max = _simulate_weight_stats(
        pred_arr, args.batch_size, fp_tau, fp_alpha, gamma, min_w, fp_max, normalize
    )

    best = None
    best_score = float("inf")
    for tau in np.linspace(0.55, 0.95, 22):
        for alpha in np.linspace(0.05, 0.65, 17):
            for max_w in np.linspace(1.08, 1.40, 14):
                m, st, mx = _simulate_weight_stats(
                    gt, args.batch_size, float(tau), float(alpha), gamma, min_w, float(max_w), normalize
                )
                err = 4.0 * (st - target_std) ** 2 + (mx - target_max) ** 2 + 2.0 * (m - target_mean) ** 2
                if err < best_score:
                    best_score = err
                    best = (float(tau), float(alpha), float(max_w), m, st, mx)

    assert best is not None
    tau_s, alpha_s, max_w_s, m_s, st_s, mx_s = best

    best2 = None
    best2_err = float("inf")
    for tau in np.linspace(0.62, 0.96, 19):
        for max_w in np.linspace(1.12, 1.35, 14):
            for alpha in np.linspace(0.02, 0.50, 90):
                m, st, mx = _simulate_weight_stats(
                    gt, args.batch_size, float(tau), float(alpha), gamma, min_w, float(max_w), normalize
                )
                err = 6.0 * (st - target_std) ** 2 + (mx - target_max) ** 2 + 3.0 * (m - target_mean) ** 2
                if err < best2_err:
                    best2_err = err
                    best2 = (float(tau), float(alpha), float(max_w), m, st, mx)
    assert best2 is not None
    tau_s2, alpha_s2, max_w_s2, m_s2, st_s2, mx_s2 = best2

    # Optional: same alpha/max as frozen probe on GT scores — scan tau to best-match std
    best3_err = float("inf")
    best3 = None
    for tau in np.linspace(0.35, 0.92, 120):
        m, st, mx = _simulate_weight_stats(
            gt, args.batch_size, float(tau), fp_alpha, gamma, min_w, fp_max, normalize
        )
        err = (st - target_std) ** 2 + 0.3 * (mx - target_max) ** 2
        if err < best3_err:
            best3_err = err
            best3 = (float(tau), fp_alpha, fp_max, m, st, mx)

    assert best3 is not None
    tau3, _, _, m3, st3, mx3 = best3
    use_same_shape = abs(st3 - target_std) < abs(st_s2 - target_std)
    rec_tau, rec_alpha, rec_max = (tau3, fp_alpha, fp_max) if use_same_shape else (tau_s2, alpha_s2, max_w_s2)
    rec_m, rec_st, rec_mx = (m3, st3, mx3) if use_same_shape else (m_s2, st_s2, mx_s2)

    out = {
        "n_train_patches": int(len(gt)),
        "tau_probe_for_trigger": float(args.tau_probe),
        "pct_probe_pred_above_tau_on_train": round(target_frac, 6),
        "matched_trigger_oracle_tau": tau_trigger,
        "matched_trigger_note": "oracle_quadratic: pct(gt_sci > tau) on train ~= pct(probe_pred > tau_probe) on train",
        "frozen_probe_weight_targets_from_train_preds": {
            "tau": fp_tau,
            "alpha": fp_alpha,
            "max_weight": fp_max,
            "pooled_mean": target_mean,
            "pooled_std": target_std,
            "pooled_max": target_max,
        },
        "matched_strength_recommended_for_p6": {
            "tau": rec_tau,
            "alpha": rec_alpha,
            "max_weight": rec_max,
            "simulated_mean": rec_m,
            "simulated_std": rec_st,
            "simulated_max": rec_mx,
            "source": "same_alpha_max_tau_scan" if use_same_shape else "joint_grid_refinement",
        },
        "matched_strength_joint_grid": {
            "tau": tau_s2,
            "alpha": alpha_s2,
            "max_weight": max_w_s2,
            "simulated_mean": m_s2,
            "simulated_std": st_s2,
            "simulated_max": mx_s2,
            "refinement_error": best2_err,
        },
        "matched_strength_same_alpha_max_tau_scan": {
            "tau": tau3,
            "alpha": fp_alpha,
            "max_weight": fp_max,
            "simulated_mean": m3,
            "simulated_std": st3,
            "simulated_max": mx3,
            "tau_scan_error": best3_err,
        },
        "matched_strength_coarse_grid": {
            "tau": tau_s,
            "alpha": alpha_s,
            "max_weight": max_w_s,
            "simulated_std": st_s,
            "simulated_max": mx_s,
        },
        "p2_reference": {"tau": 0.6, "alpha": 0.8, "max_weight": 1.8},
    }
    print(json.dumps(out, indent=2))

    out_path = Path(args.out_json)
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(out, out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
