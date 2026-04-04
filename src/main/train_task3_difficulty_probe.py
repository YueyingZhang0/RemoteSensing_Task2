#!/usr/bin/env python3
"""Train SCE probe on Task 3 *train* patches only with nested group split (no seg-val leakage)."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.datasets.patch_regression_dataset import PatchRegressionDataset
from src.models.sce_probe_cnn import SCEProbeCNN
from src.training.engine_sce_probe import fit_sce_probe
from src.training.splits import load_or_create_group_split, make_nested_group_split, save_split_info
from src.utils.io import ensure_dir, load_yaml, save_json
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train difficulty probe for Task 3 (nested split on train IDs).")
    p.add_argument("--config", type=str, default="configs/task3.yaml")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--probe_val_ratio", type=float, default=0.15, help="Group split inside seg-train only.")
    p.add_argument("--probe_random_state", type=int, default=123)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    cfg = load_yaml(cfg_path) if cfg_path.is_file() else {}
    set_global_seed(int(cfg.get("random_state", 42)))

    root = Path(__file__).resolve().parents[2]
    meta = Path(cfg["metadata_csv"])
    if not meta.is_absolute():
        meta = root / meta
    img_dir = Path(cfg["patch_image_dir"])
    if not img_dir.is_absolute():
        img_dir = root / img_dir

    target_col = cfg.get("target_col", "sci_res2_norm")
    group_col = cfg.get("group_col", "sample_id")
    val_ratio = cfg.get("val_ratio", 0.2)
    random_state = cfg.get("random_state", 42)
    split_json = cfg.get("split_json")

    df = pd.read_csv(meta)
    for c in ("patch_name", target_col, group_col):
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    if not split_json:
        raise ValueError("config.split_json is required so probe trains only on seg-train patients.")

    df_tr, df_va_seg = load_or_create_group_split(
        df, group_col, val_ratio, random_state, split_json, root,
    )

    probe_rs = int(args.probe_random_state)
    df_probe_tr, df_probe_va = make_nested_group_split(
        df_tr, group_col, float(args.probe_val_ratio), probe_rs,
    )

    out_dir = Path(args.output_dir) if args.output_dir else Path(cfg.get("output_dir_task3_probe", "outputs/task3_difficulty_probe"))
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    ensure_dir(out_dir)

    save_split_info(df_probe_tr, df_probe_va, group_col, probe_rs, float(args.probe_val_ratio), out_dir / "probe_split.json")

    seg_val_ids = sorted(df_va_seg[group_col].astype(str).unique().tolist())
    manifest = {
        "purpose": "Difficulty probe for Task 3 frozen_probe_weighted; trained only on seg-train patches.",
        "nested_split": {
            "probe_val_ratio": float(args.probe_val_ratio),
            "probe_random_state": probe_rs,
            "n_probe_train_patches": len(df_probe_tr),
            "n_probe_val_patches": len(df_probe_va),
            "n_probe_train_groups": int(df_probe_tr[group_col].nunique()),
            "n_probe_val_groups": int(df_probe_va[group_col].nunique()),
        },
        "seg_split_reference": str(Path(split_json).resolve()) if split_json else None,
        "seg_val_sample_ids": seg_val_ids,
        "note": "Probe train+val patch sample_ids are subsets of seg train; disjoint from seg_val_sample_ids by construction.",
    }
    save_json(manifest, out_dir / "probe_manifest.json")

    dev_s = cfg.get("device")
    device = torch.device(dev_s if dev_s else ("cuda" if torch.cuda.is_available() else "cpu"))
    bs = int(cfg.get("batch_size", 16))
    nw = int(cfg.get("num_workers", 0))

    train_ds = PatchRegressionDataset(df_probe_tr, img_dir, target_col, cfg.get("image_size", 128), augment=True)
    val_ds = PatchRegressionDataset(df_probe_va, img_dir, target_col, cfg.get("image_size", 128), augment=False)
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=device.type == "cuda",
    )

    model = SCEProbeCNN(in_channels=3, dropout=0.1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(cfg.get("lr", 1e-3)), weight_decay=float(cfg.get("weight_decay", 1e-4)))
    n_epochs = int(cfg.get("probe_epochs", cfg.get("epochs", 80)))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(n_epochs, 1))

    print(f"Seg-val groups (held out from probe training): {len(seg_val_ids)}")
    print(f"Probe train: {len(df_probe_tr)} patches, val: {len(df_probe_va)} patches")

    fit_sce_probe(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=n_epochs,
        patience=int(cfg.get("probe_early_stop_patience", cfg.get("early_stop_patience", 15))),
        output_dir=str(out_dir),
        target_col=target_col,
        log_epoch_metrics=False,
        plot_train_debug=False,
        plot_diagnostic_split=False,
    )
    print(f"Probe checkpoint: {out_dir / 'best_sce_branch.pt'}")


if __name__ == "__main__":
    main()
