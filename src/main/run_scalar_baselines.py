#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone entry to run scalar baselines for Task 2."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.analysis.scalar_baselines import run_scalar_regression_baselines
from src.training.splits import make_group_train_val_split, save_split_info
from src.utils.io import ensure_dir, load_yaml
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run scalar regression baselines.")
    p.add_argument("--config", type=str, default="configs/sce_probe.yaml")
    p.add_argument("--metadata_csv", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--target_col", type=str, default=None)
    p.add_argument("--group_col", type=str, default=None)
    p.add_argument("--val_ratio", type=float, default=None)
    p.add_argument("--random_state", type=int, default=None)
    return p.parse_args()


def merge_config(args: argparse.Namespace) -> dict:
    cfg_path = Path(args.config)
    cfg = load_yaml(cfg_path) if cfg_path.is_file() else {}
    for key in (
        "metadata_csv",
        "output_dir",
        "target_col",
        "group_col",
        "val_ratio",
        "random_state",
    ):
        cli_val = getattr(args, key, None)
        if cli_val is not None:
            cfg[key] = cli_val
    return cfg


def main() -> None:
    args = parse_args()
    cfg = merge_config(args)
    random_state = cfg["random_state"]
    set_global_seed(random_state)

    metadata_csv = Path(cfg["metadata_csv"])
    target_col = cfg["target_col"]
    group_col = cfg["group_col"]
    val_ratio = cfg["val_ratio"]
    output_dir = cfg["output_dir"]

    df = pd.read_csv(metadata_csv)
    for c in ["vessel_area", "fov_ratio", target_col, group_col]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    df_train, df_val = make_group_train_val_split(
        df, group_col=group_col, val_ratio=val_ratio, random_state=random_state,
    )
    out = Path(output_dir)
    ensure_dir(out)
    save_split_info(df_train, df_val, group_col, random_state, val_ratio, out / "split.json")

    print(f"Train: {len(df_train)} patches ({df_train[group_col].nunique()} images)")
    print(f"Val:   {len(df_val)} patches ({df_val[group_col].nunique()} images)")

    run_scalar_regression_baselines(df_train, df_val, target_col, output_dir)


if __name__ == "__main__":
    main()
