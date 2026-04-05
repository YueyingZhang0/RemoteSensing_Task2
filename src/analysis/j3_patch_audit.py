#!/usr/bin/env python3
"""Patch-level audit: J3 vs baseline and J3 vs P5 on seed=42 val set."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.datasets.seg_sce_patch_dataset import SegSCEPatchDataset
from src.main.train_task3 import _apply_hard_labels
from src.models.task3_unet import UNetBaseline, UNetBaselineJointDifficulty
from src.training.splits import load_or_create_group_split
from src.training.task3_engine import _per_sample_dice_recall, collate_seg_sce
from src.utils.io import load_yaml, save_json


def load_model(ckpt_path: Path, model_type: str, device: torch.device) -> torch.nn.Module:
    if model_type == "joint":
        model = UNetBaselineJointDifficulty(in_channels=3, base=32).to(device)
    else:
        model = UNetBaseline(in_channels=3, base=32).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def collect_per_patch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> pd.DataFrame:
    all_dice, all_recall, all_hard, all_names = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device)
            m = batch["mask"].to(device)
            hard = batch["is_hard"]
            out = model(img)
            logits = out[0] if isinstance(out, tuple) else out
            d, r = _per_sample_dice_recall(logits, m, thresh=threshold)
            all_dice.extend(d.tolist() if hasattr(d, "tolist") else list(d))
            all_recall.extend(r.tolist() if hasattr(r, "tolist") else list(r))
            all_hard.extend(bool(h) for h in hard)
            all_names.extend(batch.get("patch_name", [f"patch_{i}" for i in range(len(d))]))
    return pd.DataFrame({
        "patch_name": all_names,
        "dice": all_dice,
        "recall": all_recall,
        "is_hard": all_hard,
    })


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/task3_J3_joint_detach_false.yaml")
    p.add_argument("--j3_ckpt", default="outputs/task3_J3_joint_detach_false/best_model.pt")
    p.add_argument("--p0_ckpt", default="outputs/task3_v2_1_baseline_same_split/best_model.pt")
    p.add_argument("--p5_ckpt", default="outputs/task3_v3_1_oracle_matched_trigger_same_split/best_model.pt")
    p.add_argument("--out_dir", default="outputs/task3_J3_patch_audit")
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()

    root = Path(__file__).resolve().parents[2]
    cfg = load_yaml(root / args.config)
    target_col = cfg.get("target_col", "sci_res2_norm")
    group_col = cfg.get("group_col", "sample_id")
    hard_ratio = float(cfg.get("hard_ratio", 0.2))
    val_ratio = float(cfg.get("val_ratio", 0.2))
    random_state = int(cfg.get("random_state", 42))
    split_json = cfg.get("split_json")
    isz = int(cfg.get("image_size", 128))
    bs = int(cfg.get("batch_size", 16))

    meta = Path(cfg["metadata_csv"])
    if not meta.is_absolute():
        meta = root / meta
    img_dir = Path(cfg["patch_image_dir"])
    if not img_dir.is_absolute():
        img_dir = root / img_dir
    msk_dir = Path(cfg["patch_mask_dir"])
    if not msk_dir.is_absolute():
        msk_dir = root / msk_dir

    df = pd.read_csv(meta)
    df_tr, df_va = load_or_create_group_split(
        df, group_col, val_ratio, random_state, split_json, root,
    )
    df_tr, df_va = _apply_hard_labels(df_tr, df_va, target_col, hard_ratio)

    val_ds = SegSCEPatchDataset(df_va, img_dir, msk_dir, target_col, isz, augment=False)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0, collate_fn=collate_seg_sce)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading models...")
    j3_model = load_model(root / args.j3_ckpt, "joint", device)
    p0_model = load_model(root / args.p0_ckpt, "baseline", device)
    p5_model = load_model(root / args.p5_ckpt, "baseline", device)

    print("Collecting per-patch metrics...")
    j3_df = collect_per_patch(j3_model, val_loader, device, args.threshold)
    p0_df = collect_per_patch(p0_model, val_loader, device, args.threshold)
    p5_df = collect_per_patch(p5_model, val_loader, device, args.threshold)

    meta_cols = ["patch_name", target_col, "fov_ratio", "sample_id"]
    avail_cols = [c for c in meta_cols if c in df_va.columns]
    meta_sub = df_va[avail_cols].copy()

    audit = meta_sub.copy()
    audit["is_hard"] = j3_df["is_hard"].values
    audit["j3_dice"] = j3_df["dice"].values
    audit["j3_recall"] = j3_df["recall"].values
    audit["p0_dice"] = p0_df["dice"].values
    audit["p0_recall"] = p0_df["recall"].values
    audit["p5_dice"] = p5_df["dice"].values
    audit["p5_recall"] = p5_df["recall"].values
    audit["delta_j3_p0_dice"] = audit["j3_dice"] - audit["p0_dice"]
    audit["delta_j3_p0_recall"] = audit["j3_recall"] - audit["p0_recall"]
    audit["delta_j3_p5_dice"] = audit["j3_dice"] - audit["p5_dice"]
    audit["delta_j3_p5_recall"] = audit["j3_recall"] - audit["p5_recall"]

    def classify_vs(delta_col: str, threshold: float = 0.005) -> pd.Series:
        return pd.cut(
            audit[delta_col],
            bins=[-np.inf, -threshold, threshold, np.inf],
            labels=["worse", "tied", "better"],
        )

    audit["j3_vs_p0"] = classify_vs("delta_j3_p0_dice")
    audit["j3_vs_p5"] = classify_vs("delta_j3_p5_dice")

    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    audit.to_csv(out_dir / "j3_patch_audit.csv", index=False)
    print(f"Saved: {out_dir / 'j3_patch_audit.csv'}")

    def group_summary(group_col_name: str) -> List[Dict[str, Any]]:
        rows = []
        for label in ["better", "tied", "worse"]:
            sub = audit[audit[group_col_name] == label]
            row: Dict[str, Any] = {
                "group": label,
                "count": len(sub),
                "pct": round(len(sub) / len(audit) * 100, 1),
                "is_hard_pct": round(sub["is_hard"].mean() * 100, 1) if len(sub) else 0,
                "j3_dice_mean": round(float(sub["j3_dice"].mean()), 6) if len(sub) else None,
                "p0_dice_mean": round(float(sub["p0_dice"].mean()), 6) if len(sub) else None,
                "p5_dice_mean": round(float(sub["p5_dice"].mean()), 6) if len(sub) else None,
            }
            if target_col in sub.columns:
                row["sci_mean"] = round(float(sub[target_col].mean()), 4) if len(sub) else None
            if "fov_ratio" in sub.columns:
                row["fov_mean"] = round(float(sub["fov_ratio"].mean()), 4) if len(sub) else None
            rows.append(row)
        return rows

    j3_vs_p0_summary = group_summary("j3_vs_p0")
    j3_vs_p5_summary = group_summary("j3_vs_p5")

    summary = {
        "threshold": args.threshold,
        "n_val": len(audit),
        "n_hard": int(audit["is_hard"].sum()),
        "j3_vs_p0": {
            "mean_delta_dice": round(float(audit["delta_j3_p0_dice"].mean()), 6),
            "mean_delta_recall": round(float(audit["delta_j3_p0_recall"].mean()), 6),
            "groups": j3_vs_p0_summary,
        },
        "j3_vs_p5": {
            "mean_delta_dice": round(float(audit["delta_j3_p5_dice"].mean()), 6),
            "mean_delta_recall": round(float(audit["delta_j3_p5_recall"].mean()), 6),
            "groups": j3_vs_p5_summary,
        },
        "hard_subset": {
            "j3_vs_p0_mean_delta": round(float(audit[audit["is_hard"]]["delta_j3_p0_dice"].mean()), 6),
            "j3_vs_p5_mean_delta": round(float(audit[audit["is_hard"]]["delta_j3_p5_dice"].mean()), 6),
        },
    }
    save_json(summary, out_dir / "j3_patch_groups_summary.json")

    print(f"\n  J3 vs P0 (baseline):")
    for g in j3_vs_p0_summary:
        print(f"    {g['group']:>8}: {g['count']:>3} patches ({g['pct']:>5.1f}%), "
              f"hard={g['is_hard_pct']:.0f}%, sci={g.get('sci_mean','N/A')}, fov={g.get('fov_mean','N/A')}")

    print(f"\n  J3 vs P5 (oracle_matched_trigger):")
    for g in j3_vs_p5_summary:
        print(f"    {g['group']:>8}: {g['count']:>3} patches ({g['pct']:>5.1f}%), "
              f"hard={g['is_hard_pct']:.0f}%, sci={g.get('sci_mean','N/A')}, fov={g.get('fov_mean','N/A')}")

    h = audit[audit["is_hard"]]
    print(f"\n  Hard subset ({len(h)} patches):")
    print(f"    J3 vs P0 mean delta dice: {h['delta_j3_p0_dice'].mean():+.6f}")
    print(f"    J3 vs P5 mean delta dice: {h['delta_j3_p5_dice'].mean():+.6f}")

    with open(out_dir / "j3_patch_groups_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["comparison"] + list(j3_vs_p0_summary[0].keys()))
        w.writeheader()
        for g in j3_vs_p0_summary:
            g["comparison"] = "j3_vs_p0"
            w.writerow(g)
        for g in j3_vs_p5_summary:
            g["comparison"] = "j3_vs_p5"
            w.writerow(g)

    print(f"\n  Saved: {out_dir / 'j3_patch_groups_summary.json'}")
    print(f"  Saved: {out_dir / 'j3_patch_groups_summary.csv'}")


if __name__ == "__main__":
    main()
