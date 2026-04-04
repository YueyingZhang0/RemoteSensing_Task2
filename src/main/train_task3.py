#!/usr/bin/env python3
"""Task 3 minimal: U-Net baseline or UNet+SCE (sci_res2_norm fixed)."""
from __future__ import annotations

import argparse
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.datasets.seg_sce_patch_dataset import SegSCEPatchDataset
from src.models.task3_unet import UNetBaseline, UNetWithSCE
from src.training.splits import load_or_create_group_split, make_group_train_val_split, mark_hard_patches
from src.training.task3_engine import (
    collate_seg_sce,
    train_task3,
    train_task3_baseline_oracle_weighted,
)
from src.utils.io import ensure_dir, load_yaml
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task 3: U-Net seg (+ SCE ours).")
    p.add_argument("--config", type=str, default="configs/task3.yaml")
    p.add_argument("--model", type=str, choices=("baseline", "ours"), required=True)
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output_dir_baseline / output_dir_ours for this run (relative paths from project root).",
    )
    return p.parse_args()


def _apply_hard_labels(df_tr: pd.DataFrame, df_va: pd.DataFrame, target_col: str, hard_ratio: float):
    df_va = mark_hard_patches(df_va, target_col, hard_ratio)
    thr = float(np.quantile(df_va[target_col].values, 1.0 - hard_ratio))
    df_tr = df_tr.copy()
    df_tr["is_hard"] = df_tr[target_col] >= thr
    return df_tr, df_va


def _resolve_split_json_path(split_json: str | None, project_root: Path) -> str | None:
    if not split_json:
        return None
    p = Path(split_json)
    if not p.is_absolute():
        p = project_root / p
    return str(p.resolve())


def _git_commit_hash(project_root: Path) -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode != 0:
            return None
        h = (r.stdout or "").strip()
        return h or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _dump_config_used(out_dir: Path, doc: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True)


def _build_config_used(
    cfg: Dict[str, Any],
    *,
    project_root: Path,
    out_dir: Path,
    config_path: Path,
    metadata_csv: Path,
    patch_image_dir: Path,
    patch_mask_dir: Path,
    model: str,
    warmup_epochs: int,
    joint_epochs: int,
    difficulty_weighting_effective: Dict[str, Any],
) -> Dict[str, Any]:
    doc = deepcopy(cfg)
    doc["model"] = model
    doc["output_dir"] = str(out_dir.resolve())
    doc["split_json"] = _resolve_split_json_path(cfg.get("split_json"), project_root)
    doc["metadata_csv"] = str(metadata_csv.resolve())
    doc["patch_image_dir"] = str(patch_image_dir.resolve())
    doc["patch_mask_dir"] = str(patch_mask_dir.resolve())
    doc["epochs"] = int(warmup_epochs + joint_epochs)
    doc["warmup_epochs"] = int(warmup_epochs)
    doc["joint_epochs"] = int(joint_epochs)
    doc["batch_size"] = int(cfg.get("batch_size", 16))
    doc["lr"] = float(cfg.get("lr", 1e-3))
    tc = cfg.get("target_col", "sci_res2_norm")
    hr = float(cfg.get("hard_ratio", 0.2))
    doc["hard_patch"] = {
        "target_col": tc,
        "hard_ratio": hr,
        "definition": (
            "Mark hardest hard_ratio fraction of val patches by target_col (quantile threshold); "
            "train is_hard uses the same threshold."
        ),
    }
    doc["difficulty_weighting"] = difficulty_weighting_effective
    doc["config_file"] = str(config_path.resolve())
    doc["cli_args"] = list(sys.argv)
    doc["git_commit"] = _git_commit_hash(project_root)
    doc["timestamp"] = datetime.now(timezone.utc).isoformat()
    return doc


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
    val_ratio = cfg.get("val_ratio", 0.2)
    random_state = cfg.get("random_state", 42)
    hard_ratio = cfg.get("hard_ratio", 0.2)
    split_json = cfg.get("split_json")

    df = pd.read_csv(meta)
    for c in ("patch_name", target_col, group_col):
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    if split_json:
        df_tr, df_va = load_or_create_group_split(
            df, group_col, val_ratio, random_state, split_json, root,
        )
    else:
        df_tr, df_va = make_group_train_val_split(df, group_col, val_ratio, random_state)

    df_tr, df_va = _apply_hard_labels(df_tr, df_va, target_col, hard_ratio)

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

    dw = cfg.get("difficulty_weighting") or {}
    if dw.get("enabled"):
        if args.model != "baseline":
            raise ValueError("difficulty_weighting.enabled requires --model baseline (no SCE v2 path).")
        out_dir = Path(args.output_dir) if args.output_dir else Path(cfg["output_dir"])
        if not out_dir.is_absolute():
            out_dir = root / out_dir
        ensure_dir(out_dir)

        log_cfg = cfg.get("logging") or {}
        w_ep = int(cfg.get("warmup_epochs", 30))
        j_ep = int(cfg.get("joint_epochs", 50))
        dw_eff: Dict[str, Any] = {
            "enabled": True,
            "tau": float(dw.get("tau", 0.7)),
            "alpha": float(dw.get("alpha", 1.0)),
            "gamma": float(dw.get("gamma", 2.0)),
            "weight_warmup_epochs": int(dw.get("warmup_epochs", 5)),
            "min_weight": float(dw.get("min_weight", 1.0)),
            "max_weight": float(dw.get("max_weight", 2.0)),
            "normalize_weights_in_batch": bool(dw.get("normalize_weights_in_batch", True)),
            "save_weight_stats": bool(log_cfg.get("save_weight_stats", True)),
        }
        _dump_config_used(
            out_dir,
            _build_config_used(
                cfg,
                project_root=root,
                out_dir=out_dir,
                config_path=cfg_path,
                metadata_csv=meta,
                patch_image_dir=img_dir,
                patch_mask_dir=msk_dir,
                model="baseline",
                warmup_epochs=w_ep,
                joint_epochs=j_ep,
                difficulty_weighting_effective=dw_eff,
            ),
        )
        train_task3_baseline_oracle_weighted(
            model=UNetBaseline(in_channels=3, base=32).to(device),
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            out_dir=out_dir,
            total_epochs=w_ep + j_ep,
            lr=float(cfg.get("lr", 1e-3)),
            weight_decay=float(cfg.get("weight_decay", 1e-4)),
            tau=float(dw.get("tau", 0.7)),
            alpha=float(dw.get("alpha", 1.0)),
            gamma=float(dw.get("gamma", 2.0)),
            difficulty_warmup_epochs=int(dw.get("warmup_epochs", 5)),
            min_weight=float(dw.get("min_weight", 1.0)),
            max_weight=float(dw.get("max_weight", 2.0)),
            normalize_weights_in_batch=bool(dw.get("normalize_weights_in_batch", True)),
            save_weight_stats=bool(log_cfg.get("save_weight_stats", True)),
        )
        print(f"Done (oracle-weighted baseline). Artifacts in {out_dir}")
        return

    out_key = "output_dir_baseline" if args.model == "baseline" else "output_dir_ours"
    out_dir = Path(args.output_dir) if args.output_dir else Path(cfg[out_key])
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    ensure_dir(out_dir)

    w_ep = int(cfg.get("warmup_epochs", 30))
    j_ep = int(cfg.get("joint_epochs", 50))
    raw_dw = cfg.get("difficulty_weighting")
    dw_eff: Dict[str, Any] = {"enabled": False}
    if raw_dw is not None:
        dw_eff["config_as_loaded"] = deepcopy(raw_dw)
    _dump_config_used(
        out_dir,
        _build_config_used(
            cfg,
            project_root=root,
            out_dir=out_dir,
            config_path=cfg_path,
            metadata_csv=meta,
            patch_image_dir=img_dir,
            patch_mask_dir=msk_dir,
            model=args.model,
            warmup_epochs=w_ep,
            joint_epochs=j_ep,
            difficulty_weighting_effective=dw_eff,
        ),
    )

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
        warmup_epochs=w_ep,
        joint_epochs=j_ep,
        lr=float(cfg.get("lr", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
        lambda_sce=float(cfg.get("lambda_sce", 0.5)),
    )
    print(f"Done. Artifacts in {out_dir}")


if __name__ == "__main__":
    main()
