#!/usr/bin/env python3
"""Task 3 minimal: U-Net baseline or UNet+SCE (sci_res2_norm fixed)."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.datasets.seg_sce_patch_dataset import SegSCEPatchDataset
from src.models.task3_unet import UNetBaseline, UNetWithSCE
from src.training.splits import make_group_train_val_split, mark_hard_patches
from src.training.task3_engine import collate_seg_sce, train_task3
from src.utils.io import ensure_dir, load_yaml
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task 3: U-Net seg (+ SCE ours).")
    p.add_argument("--config", type=str, default="configs/task3.yaml")
    p.add_argument("--model", type=str, choices=("baseline", "ours"), required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    cfg = load_yaml(cfg_path) if cfg_path.is_file() else {}
    set_global_seed(cfg.get("random_state", 42))

    root = Path(__file__).resolve().parents[2]
    meta = Path(cfg["metadata_csv"])
    if not meta.is_absolute():
        meta = root / meta
    img_dir = Path(cfg["patch_image_dir"])
    if not img_dir.is_absolute():
        img_dir = root / img_dir
    msk_dir = Path(cfg["patch_mask_dir"])
    if not msk_dir.is_absolute():
        msk_dir = root / msk_dir

    target_col = cfg.get("target_col", "sci_res2_norm")
    group_col = cfg.get("group_col", "sample_id")
    out_key = "output_dir_baseline" if args.model == "baseline" else "output_dir_ours"
    out_dir = Path(cfg[out_key])
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    ensure_dir(out_dir)

    df = pd.read_csv(meta)
    for c in ("patch_name", target_col, group_col):
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    df_tr, df_va = make_group_train_val_split(
        df, group_col, cfg.get("val_ratio", 0.2), cfg.get("random_state", 42),
    )
    df_va = mark_hard_patches(df_va, target_col, cfg.get("hard_ratio", 0.2))

    bs = cfg.get("batch_size", 16)
    nw = cfg.get("num_workers", 0)
    isz = cfg.get("image_size", 128)

    train_ds = SegSCEPatchDataset(df_tr, img_dir, msk_dir, target_col, isz, augment=True)
    val_ds = SegSCEPatchDataset(df_va, img_dir, msk_dir, target_col, isz, augment=False)
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=nw, collate_fn=collate_seg_sce,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=nw, collate_fn=collate_seg_sce,
        pin_memory=torch.cuda.is_available(),
    )

    dev_s = cfg.get("device")
    device = torch.device(dev_s if dev_s else ("cuda" if torch.cuda.is_available() else "cpu"))

    if args.model == "baseline":
        model = UNetBaseline(in_channels=3, base=32).to(device)
        ours = False
    else:
        model = UNetWithSCE(in_channels=3, base=32).to(device)
        ours = True

    train_task3(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        ours=ours,
        out_dir=out_dir,
        warmup_epochs=int(cfg.get("warmup_epochs", 30)),
        joint_epochs=int(cfg.get("joint_epochs", 50)),
        lr=float(cfg.get("lr", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
        lambda_sce=float(cfg.get("lambda_sce", 0.5)),
    )
    print(f"Done. Artifacts in {out_dir}")


if __name__ == "__main__":
    main()
