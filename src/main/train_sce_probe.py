#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 2 main entry: train SCE probe CNN, run scalar baselines, generate analysis.
Usage:  python -m src.main.train_sce_probe [--config configs/sce_probe.yaml] ...
        python -m src.main.train_sce_probe --debug-overfit --output_dir outputs/task2_debug_overfit ...
        python -m src.main.train_sce_probe --diagnostic-full ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.analysis.prediction_reports import (
    generate_analysis_report,
    save_top_bottom_predictions,
)
from src.analysis.scalar_baselines import run_scalar_regression_baselines
from src.datasets.patch_regression_dataset import PatchRegressionDataset
from src.models.sce_probe_cnn import SCEProbeCNN
from src.training.engine_sce_probe import fit_sce_probe
from src.training.splits import make_group_train_val_split, save_split_info
from src.utils.io import load_yaml, ensure_dir
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SCE probe (Task 2).")
    p.add_argument("--config", type=str, default="configs/sce_probe.yaml")
    p.add_argument("--metadata_csv", type=str, default=None)
    p.add_argument("--patch_image_dir", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--target_col", type=str, default=None)
    p.add_argument("--group_col", type=str, default=None)
    p.add_argument("--val_ratio", type=float, default=None)
    p.add_argument("--random_state", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--early_stop_patience", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--skip_baselines", action="store_true")
    p.add_argument(
        "--debug-overfit",
        action="store_true",
        help="32 patches, same train/val, no aug, dropout 0, 200 epochs, train-only debug plots.",
    )
    p.add_argument(
        "--diagnostic-full",
        action="store_true",
        help="Normal sample_id split; log train/val Pearson+Spearman each epoch; extra val/train plots.",
    )
    return p.parse_args()


def merge_config(args: argparse.Namespace) -> dict:
    cfg_path = Path(args.config)
    cfg = load_yaml(cfg_path) if cfg_path.is_file() else {}
    for key in [
        "metadata_csv", "patch_image_dir", "output_dir", "target_col",
        "group_col", "val_ratio", "random_state", "batch_size", "epochs",
        "lr", "weight_decay", "early_stop_patience", "num_workers", "device",
    ]:
        cli_val = getattr(args, key, None)
        if cli_val is not None:
            cfg[key] = cli_val
    if "device" not in cfg or cfg["device"] is None:
        cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg


def main() -> None:
    args = parse_args()
    if args.debug_overfit and args.diagnostic_full:
        raise ValueError("Use only one of --debug-overfit or --diagnostic-full.")

    cfg = merge_config(args)

    set_global_seed(cfg.get("random_state", 42))

    metadata_csv = Path(cfg["metadata_csv"])
    image_dir = Path(cfg["patch_image_dir"])
    out_dir = Path(cfg["output_dir"])
    ensure_dir(out_dir)

    if not metadata_csv.is_file():
        raise FileNotFoundError(f"metadata_csv not found: {metadata_csv}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"patch_image_dir not found: {image_dir}")

    group_col = cfg["group_col"]
    val_ratio = cfg["val_ratio"]
    random_state = cfg["random_state"]

    df = pd.read_csv(metadata_csv)
    for c in ("patch_name", group_col):
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    log_epoch_metrics = False
    plot_train_debug = False
    plot_diagnostic_split = False
    skip_analysis_tail = False
    dropout_p = 0.1
    augment_train = True

    if args.debug_overfit:
        target_col = "sci_res2_norm"
        cfg["target_col"] = target_col
        if target_col not in df.columns:
            raise ValueError(f"Missing column for debug: {target_col}")
        n_dbg = min(32, len(df))
        df_sub = df.sample(n=n_dbg, random_state=random_state).reset_index(drop=True)
        df_train = df_val = df_sub
        cfg["epochs"] = 200
        cfg["early_stop_patience"] = max(int(cfg.get("early_stop_patience", 15)), 500)
        cfg["weight_decay"] = 0.0
        augment_train = False
        dropout_p = 0.0
        plot_train_debug = True
        skip_analysis_tail = True
        print("\n=== DEBUG OVERFIT mode: 32 patches (or fewer if CSV smaller), same train/val, target=sci_res2_norm ===\n")
    else:
        target_col = cfg["target_col"]
        if target_col not in df.columns:
            raise ValueError(f"Missing column: {target_col}")
        df_train, df_val = make_group_train_val_split(df, group_col, val_ratio, random_state)
        if args.diagnostic_full:
            log_epoch_metrics = True
            plot_diagnostic_split = True
            print("\n=== DIAGNOSTIC-FULL: per-epoch train/val correlations + split plots ===\n")

    save_split_info(df_train, df_val, group_col, random_state, val_ratio, out_dir / "split.json")

    print(f"Train: {len(df_train)} patches ({df_train[group_col].nunique()} images)")
    print(f"Val:   {len(df_val)} patches ({df_val[group_col].nunique()} images)")

    if not args.skip_baselines:
        print("\n--- Scalar baselines ---")
        run_scalar_regression_baselines(df_train, df_val, target_col, str(out_dir))

    print("\n--- SCE probe CNN ---")
    device = torch.device(cfg["device"])
    batch_size = cfg.get("batch_size", 32)
    num_workers = cfg.get("num_workers", 0)

    train_ds = PatchRegressionDataset(df_train, image_dir, target_col, augment=augment_train)
    val_ds = PatchRegressionDataset(df_val, image_dir, target_col, augment=False)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=device.type == "cuda",
    )

    model = SCEProbeCNN(in_channels=cfg.get("in_channels", 3), dropout=dropout_p).to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=cfg.get("lr", 1e-3), weight_decay=cfg.get("weight_decay", 1e-4),
    )
    n_epochs = int(cfg.get("epochs", 100))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    result = fit_sce_probe(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=n_epochs,
        patience=int(cfg.get("early_stop_patience", 15)),
        output_dir=str(out_dir),
        target_col=target_col,
        log_epoch_metrics=log_epoch_metrics,
        plot_train_debug=plot_train_debug,
        plot_diagnostic_split=plot_diagnostic_split,
    )

    va = result["val_result"]
    tr_metrics = result["metrics"]["train"]
    print(f"\nBest epoch: {result['best_epoch']}")
    print(f"Train loss={tr_metrics['loss']:.6f}  Pearson={tr_metrics['pearson']:.4f}  Spearman={tr_metrics['spearman']:.4f}")
    print(f"Val   loss={va['loss']:.6f}  Pearson={va['pearson']:.4f}  Spearman={va['spearman']:.4f}")

    if args.debug_overfit:
        print("\n=== DEBUG OVERFIT sanity ===")
        print(f"  train_loss={tr_metrics['loss']:.6f}")
        print(f"  train_Pearson={tr_metrics['pearson']:.4f}")
        print(f"  train_Spearman={tr_metrics['spearman']:.4f}")
        if tr_metrics["pearson"] < 0.85:
            print(
                "  WARNING: train Pearson < 0.85 on <=32 samples with no aug/dropout — "
                "suspect implementation/optimization before blaming the target.",
            )
        print(f"  See pred_vs_gt_train.png, y_true_hist_train.png, y_pred_hist_train.png under {out_dir}")

    if skip_analysis_tail:
        print(f"\nSkipping analysis tail (debug-overfit). Outputs in: {out_dir}")
        return

    save_top_bottom_predictions(
        va["patch_names"], va["y_pred"], df_val, out_dir / "highest_pred.txt", 20, descending=True,
        target_col=target_col,
    )
    save_top_bottom_predictions(
        va["patch_names"], va["y_pred"], df_val, out_dir / "lowest_pred.txt", 20, descending=False,
        target_col=target_col,
    )

    cnn_m = {k: v for k, v in va.items() if k not in ("y_true", "y_pred", "patch_names")}
    generate_analysis_report(
        cnn_metrics=cnn_m,
        baseline_json_path=out_dir / "baseline_results.json",
        output_dir=str(out_dir),
    )

    print(f"\nAll outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
