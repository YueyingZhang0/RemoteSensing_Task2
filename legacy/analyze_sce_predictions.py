#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Re-evaluate SCE probe on validation split, refresh plots/lists, and compare to scalar baselines.
Uses split.json from train_sce_probe output for reproducible image-level val set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

from train_sce_probe import PatchSCIDataset, SCEProbe, evaluate


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    pearson = pearsonr(y_true, y_pred)[0] if len(y_true) > 1 else float("nan")
    spearman = spearmanr(y_true, y_pred)[0] if len(y_true) > 1 else float("nan")
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return {
        "pearson": float(pearson),
        "spearman": float(spearman),
        "mae": float(mae),
        "rmse": rmse,
        "r2": float(r2),
    }


def save_scatter_gt_pred(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, title: str, target_col: str) -> None:
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, alpha=0.5, s=12)
    lo = min(float(np.min(y_true)), float(np.min(y_pred)))
    hi = max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.plot([lo, hi], [lo, hi], linestyle="--", color="gray")
    plt.xlabel(f"GT {target_col}")
    plt.ylabel(f"Pred {target_col}")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def write_ranked_predictions(
    names: List[str],
    preds: np.ndarray,
    df_meta: pd.DataFrame,
    out_path: Path,
    descending: bool,
    top_k: int,
) -> None:
    order = np.argsort(-preds if descending else preds)
    lines = []
    for i in order[:top_k]:
        name = names[int(i)]
        p = float(preds[int(i)])
        sub = df_meta[df_meta["patch_name"] == name]
        if len(sub) == 0:
            lines.append(f"{name}\tpred={p:.6f}\t(no metadata row)\n")
            continue
        r = sub.iloc[0]
        tgt = float(r.get("sci_res2_norm", float("nan")))
        lines.append(
            f"{name}\tpred={p:.6f}\tgt={tgt:.6f}\t"
            f"vessel_area={int(r['vessel_area'])}\tfov_ratio={float(r['fov_ratio']):.4f}\t"
            f"junctions={int(r['junction_count'])}\tend_eff={int(r['endpoint_eff'])}\n"
        )
    out_path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_csv", type=str, default="outputs/task1_topo_v3/patch_metadata.csv")
    parser.add_argument("--images_dir", type=str, default="outputs/task1_topo_v3/patches/images")
    parser.add_argument("--probe_dir", type=str, default="outputs/task2_sce_probe")
    parser.add_argument("--baseline_json", type=str, default="outputs/task2_sce_probe/baseline_results.json")
    parser.add_argument("--target_col", type=str, default="sci_res2_norm")
    parser.add_argument("--group_col", type=str, default="sample_id")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pearson_min", type=float, default=0.35)
    parser.add_argument("--spearman_min", type=float, default=0.40)
    parser.add_argument("--pearson_margin", type=float, default=0.10)
    args = parser.parse_args()

    probe_dir = Path(args.probe_dir)
    split_path = probe_dir / "split.json"
    ckpt_path = probe_dir / "best_sce_branch.pt"
    if not split_path.is_file():
        raise FileNotFoundError(f"Missing {split_path}; run train_sce_probe.py first.")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing {ckpt_path}; run train_sce_probe.py first.")

    split_info = json.loads(split_path.read_text(encoding="utf-8"))
    val_ids = set(split_info["val_sample_ids"])

    df = pd.read_csv(args.metadata_csv)
    df_val = df[df[args.group_col].isin(val_ids)].reset_index(drop=True)
    if len(df_val) == 0:
        raise RuntimeError("Validation split is empty; check split.json vs metadata.")

    device = torch.device(args.device)
    model = SCEProbe().to(device)
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    val_ds = PatchSCIDataset(df_val, Path(args.images_dir), args.target_col, augment=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    criterion = nn.SmoothL1Loss()
    val_loss, pred_val, gt_val, names_val = evaluate(model, val_loader, criterion, device)
    cnn_metrics = compute_metrics(gt_val, pred_val)
    cnn_metrics["loss_smooth_l1"] = val_loss

    out_dir = probe_dir
    save_scatter_gt_pred(gt_val, pred_val, out_dir / "best_scatter.png", "SCE probe: val", args.target_col)
    save_scatter_gt_pred(gt_val, pred_val, out_dir / "pred_vs_gt.png", "SCE probe: val pred vs GT", args.target_col)
    write_ranked_predictions(names_val, pred_val, df_val, out_dir / "highest_pred.txt", True, 20)
    write_ranked_predictions(names_val, pred_val, df_val, out_dir / "lowest_pred.txt", False, 20)

    analysis = {
        "cnn_val": cnn_metrics,
        "thresholds": {
            "pearson_min": args.pearson_min,
            "spearman_min": args.spearman_min,
            "pearson_margin_over_best_scalar": args.pearson_margin,
        },
    }

    baseline_path = Path(args.baseline_json)
    scalar_pearsons: Dict[str, float] = {}
    if baseline_path.is_file():
        baselines = json.loads(baseline_path.read_text(encoding="utf-8"))
        analysis["scalar_baselines"] = baselines
        for name, item in baselines.items():
            if isinstance(item, dict) and "val" in item and "pearson" in item["val"]:
                scalar_pearsons[name] = float(item["val"]["pearson"])
    else:
        analysis["scalar_baselines"] = None
        print(f"Warning: baseline JSON not found at {baseline_path}; run baseline_scalar_regressors.py with same split.")

    best_scalar_pearson = max(scalar_pearsons.values()) if scalar_pearsons else float("-inf")
    analysis["best_scalar_val_pearson"] = best_scalar_pearson if scalar_pearsons else None

    pass_pearson = cnn_metrics["pearson"] > args.pearson_min
    pass_spearman = cnn_metrics["spearman"] > args.spearman_min
    pass_margin = cnn_metrics["pearson"] > best_scalar_pearson + args.pearson_margin if scalar_pearsons else True

    analysis["verdict"] = {
        "pass_pearson": pass_pearson,
        "pass_spearman": pass_spearman,
        "pass_pearson_margin_over_scalar": pass_margin,
        "pass_overall": bool(pass_pearson and pass_spearman and pass_margin),
    }

    (out_dir / "analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")

    print("=== SCE probe analysis (validation) ===")
    print(f"SmoothL1 loss: {val_loss:.6f}")
    print(
        f"CNN  Pearson={cnn_metrics['pearson']:.4f} Spearman={cnn_metrics['spearman']:.4f} "
        f"MAE={cnn_metrics['mae']:.4f} RMSE={cnn_metrics['rmse']:.4f} R2={cnn_metrics['r2']:.4f}"
    )
    if scalar_pearsons:
        print("\nScalar baseline val Pearson:")
        for k, v in sorted(scalar_pearsons.items(), key=lambda x: -x[1]):
            print(f"  {k:<22s} {v:>8.4f}")
        print(f"\nBest scalar Pearson: {best_scalar_pearson:.4f}")
        print(f"CNN - best scalar:   {cnn_metrics['pearson'] - best_scalar_pearson:+.4f}")

    print("\n=== Acceptance (Task 2) ===")
    print(f"  val Pearson > {args.pearson_min}: {pass_pearson}")
    print(f"  val Spearman > {args.spearman_min}: {pass_spearman}")
    print(f"  CNN Pearson > best_scalar + {args.pearson_margin}: {pass_margin}")
    print(f"  OVERALL PASS: {analysis['verdict']['pass_overall']}")


if __name__ == "__main__":
    main()
