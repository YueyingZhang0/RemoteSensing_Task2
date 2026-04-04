#!/usr/bin/env python3
"""
Step C: same CNN, same sample_id split, same training config; only target_col changes.
Compares sci_topo_norm, sci_res_norm, sci_res2_norm — metrics + strict/relaxed verdicts.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.analysis.prediction_reports import generate_analysis_report, save_top_bottom_predictions
from src.analysis.scalar_baselines import run_scalar_regression_baselines
from src.datasets.patch_regression_dataset import PatchRegressionDataset
from src.models.sce_probe_cnn import SCEProbeCNN
from src.training.engine_sce_probe import fit_sce_probe
from src.training.splits import make_group_train_val_split, save_split_info
from src.utils.io import load_json, load_yaml, ensure_dir, save_json
from src.utils.seed import set_global_seed

TARGETS = ("sci_topo_norm", "sci_res_norm", "sci_res2_norm")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task 2 target comparison (same split, three targets).")
    p.add_argument("--config", type=str, default="configs/sce_probe.yaml")
    p.add_argument("--output_base", type=str, default=None, help="Parent dir for per-target runs (default: cfg output_dir + _target_cmp)")
    p.add_argument("--skip_baselines", action="store_true")
    return p.parse_args()


def merge_cfg(args: argparse.Namespace) -> dict:
    cfg_path = Path(args.config)
    cfg = load_yaml(cfg_path) if cfg_path.is_file() else {}
    return cfg


def main() -> None:
    args = parse_args()
    cfg = merge_cfg(args)
    set_global_seed(cfg.get("random_state", 42))

    metadata_csv = Path(cfg["metadata_csv"])
    image_dir = Path(cfg["patch_image_dir"])
    if not metadata_csv.is_file():
        raise FileNotFoundError(metadata_csv)
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)

    group_col = cfg["group_col"]
    val_ratio = cfg["val_ratio"]
    random_state = cfg["random_state"]

    df = pd.read_csv(metadata_csv)
    for t in TARGETS:
        if t not in df.columns:
            raise ValueError(f"Missing column {t} in metadata CSV")
    for c in ("patch_name", group_col):
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    df_train, df_val = make_group_train_val_split(df, group_col, val_ratio, random_state)

    if args.output_base:
        base = Path(args.output_base)
    else:
        base = Path(cfg.get("output_dir", "outputs/task2_sce_probe")).parent / "task2_target_cmp"
    if not base.is_absolute():
        base = Path.cwd() / base
    ensure_dir(base)
    save_split_info(df_train, df_val, group_col, random_state, val_ratio, base / "split.json")

    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    batch_size = cfg.get("batch_size", 32)
    num_workers = cfg.get("num_workers", 0)
    n_epochs = int(cfg.get("epochs", 100))
    patience = int(cfg.get("early_stop_patience", 15))

    summary: dict = {"split_parent": str(base.resolve()), "targets": {}}

    for target_col in TARGETS:
        out_dir = base / target_col
        ensure_dir(out_dir)
        print(f"\n========== Target: {target_col} -> {out_dir} ==========")

        if not args.skip_baselines:
            run_scalar_regression_baselines(df_train, df_val, target_col, str(out_dir))

        train_ds = PatchRegressionDataset(df_train, image_dir, target_col, augment=True)
        val_ds = PatchRegressionDataset(df_val, image_dir, target_col, augment=False)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=device.type == "cuda",
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=device.type == "cuda",
        )

        model = SCEProbeCNN(in_channels=cfg.get("in_channels", 3), dropout=0.1).to(device)
        optimizer = optim.Adam(
            model.parameters(), lr=cfg.get("lr", 1e-3), weight_decay=cfg.get("weight_decay", 1e-4),
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

        result = fit_sce_probe(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epochs=n_epochs,
            patience=patience,
            output_dir=str(out_dir),
            target_col=target_col,
            log_epoch_metrics=False,
            plot_train_debug=False,
            plot_diagnostic_split=False,
        )

        va = result["val_result"]
        save_top_bottom_predictions(
            va["patch_names"], va["y_pred"], df_val, out_dir / "highest_pred.txt", 20, True, target_col=target_col,
        )
        save_top_bottom_predictions(
            va["patch_names"], va["y_pred"], df_val, out_dir / "lowest_pred.txt", 20, False, target_col=target_col,
        )

        cnn_m = {k: v for k, v in va.items() if k not in ("y_true", "y_pred", "patch_names")}
        analysis = generate_analysis_report(
            cnn_metrics=cnn_m,
            baseline_json_path=out_dir / "baseline_results.json",
            output_dir=str(out_dir),
        )

        metrics = load_json(out_dir / "metrics.json")
        best_ep = metrics["config"]["best_epoch"]
        val_p = float(metrics["val"]["pearson"])
        val_s = float(metrics["val"]["spearman"])
        best_scalar = analysis.get("best_scalar_val_pearson")
        margin = None if best_scalar is None else val_p - float(best_scalar)

        summary["targets"][target_col] = {
            "output_dir": str(out_dir.resolve()),
            "best_epoch": best_ep,
            "val_pearson": val_p,
            "val_spearman": val_s,
            "best_scalar_val_pearson": best_scalar,
            "cnn_minus_best_scalar_pearson": margin,
            "verdict_pass_overall": analysis["verdict"]["pass_overall"],
            "verdict_pass_relaxed": analysis["verdict_task2_relaxed"]["pass_relaxed"],
        }

        print(f"  best_epoch={best_ep}  val_Pearson={val_p:.4f}  val_Spearman={val_s:.4f}")
        print(f"  strict={analysis['verdict']['pass_overall']}  relaxed={analysis['verdict_task2_relaxed']['pass_relaxed']}")

    save_json(summary, base / "target_comparison_summary.json")
    print(f"\nWrote {base / 'target_comparison_summary.json'}")


if __name__ == "__main__":
    main()
