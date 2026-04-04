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
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.datasets.seg_sce_patch_dataset import SegSCEPatchDataset
from src.models.task3_unet import UNetBaseline, UNetBaselineJointDifficulty, UNetWithSCE
from src.training.splits import load_or_create_group_split, make_group_train_val_split, mark_hard_patches
from src.training.task3_engine import (
    _effective_oracle_score_gamma,
    collate_seg_sce,
    load_frozen_sce_probe,
    train_task3,
    train_task3_baseline_oracle_weighted,
    train_task3_baseline_probe_weighted,
    train_task3_joint_difficulty,
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
    p.add_argument(
        "--random_state",
        type=int,
        default=None,
        help="Override config random_state (DataLoader/model init via set_global_seed).",
    )
    p.add_argument(
        "--shuffle_score_ablation",
        action="store_true",
        help="When difficulty_weighting is enabled (oracle mode), shuffle GT scores within each batch (ablation).",
    )
    p.add_argument(
        "--threshold_sweep",
        action="store_true",
        help="Force-run validation threshold sweep (uses config threshold_sweep.thresholds).",
    )
    p.add_argument(
        "--weight_mode",
        type=str,
        default=None,
        choices=("oracle_linear", "oracle_smart", "oracle_true_linear", "oracle_quadratic"),
        help="Override difficulty_weighting.weight_mode (oracle branch only).",
    )
    p.add_argument(
        "--use_weighted_sampler",
        action="store_true",
        help="Override sampling.use_weighted_sampler to true (oracle branch only).",
    )
    return p.parse_args()


def _build_weighted_sampler_weights(
    df_tr: pd.DataFrame,
    target_col: str,
    tau: float,
    fov_min: float,
    hard_sampling_weight: float,
) -> WeightedRandomSampler:
    """Oversample hard patches by score only: s > tau (no FOV gate). fov_min kept for API/config compat."""
    _ = fov_min  # legacy; difficulty_weighting.fov_min no longer affects sampler mask
    s = df_tr[target_col].astype(np.float64).values
    weights = np.ones(len(df_tr), dtype=np.float64)
    hard_mask = s > tau
    weights[hard_mask] = float(hard_sampling_weight)
    w_t = torch.from_numpy(weights).double()
    return WeightedRandomSampler(w_t, num_samples=len(df_tr), replacement=True)


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
    if args.random_state is not None:
        cfg["random_state"] = int(args.random_state)
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

    dw_pre = cfg.get("difficulty_weighting") or {}
    sam_pre = cfg.get("sampling") or {}
    use_weighted_sampler = bool(sam_pre.get("use_weighted_sampler", False)) or bool(args.use_weighted_sampler)
    use_weighted_sampler = use_weighted_sampler and bool(dw_pre.get("enabled")) and str(dw_pre.get("mode", "oracle")).lower() == "oracle"

    pin = torch.cuda.is_available()
    if use_weighted_sampler:
        tau_s = float(dw_pre.get("tau", 0.7))
        fov_min_s = float(dw_pre.get("fov_min", 0.8))
        hsw = float(sam_pre.get("hard_sampling_weight", 2.0))
        sampler = _build_weighted_sampler_weights(df_tr, target_col, tau_s, fov_min_s, hsw)
        train_loader = DataLoader(
            train_ds,
            batch_size=bs,
            sampler=sampler,
            shuffle=False,
            num_workers=nw,
            collate_fn=collate_seg_sce,
            pin_memory=pin,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=bs, shuffle=True, num_workers=nw, collate_fn=collate_seg_sce,
            pin_memory=pin,
        )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=nw, collate_fn=collate_seg_sce,
        pin_memory=pin,
    )

    dev_s = cfg.get("device")
    device = torch.device(dev_s if dev_s else ("cuda" if torch.cuda.is_available() else "cpu"))

    dw = cfg.get("difficulty_weighting") or {}
    if dw.get("enabled"):
        if args.model != "baseline":
            raise ValueError("difficulty_weighting.enabled requires --model baseline (no SCE v1 path).")
        mode = str(dw.get("mode", "oracle")).lower()
        if mode not in ("oracle", "frozen_probe", "joint"):
            raise ValueError("difficulty_weighting.mode must be oracle | frozen_probe | joint")
        out_dir = Path(args.output_dir) if args.output_dir else Path(cfg["output_dir"])
        if not out_dir.is_absolute():
            out_dir = root / out_dir
        ensure_dir(out_dir)

        log_cfg = cfg.get("logging") or {}
        w_ep = int(cfg.get("warmup_epochs", 30))
        j_ep = int(cfg.get("joint_epochs", 50))
        total_ep = w_ep + j_ep
        shuffle_ab = bool(dw.get("shuffle_scores_in_batch", False)) or bool(args.shuffle_score_ablation)
        weight_mode = (args.weight_mode or str(dw.get("weight_mode", "oracle_linear"))).lower()
        if weight_mode not in ("oracle_linear", "oracle_smart", "oracle_true_linear", "oracle_quadratic"):
            raise ValueError(
                "difficulty_weighting.weight_mode must be "
                "oracle_linear | oracle_true_linear | oracle_quadratic | oracle_smart"
            )
        fov_min_v = float(dw.get("fov_min", 0.8))
        gamma_cfg = float(dw.get("gamma", 2.0))
        gamma_effective = _effective_oracle_score_gamma(weight_mode, gamma_cfg)
        ts_cfg = cfg.get("threshold_sweep") or {}
        sweep_on = bool(ts_cfg.get("enabled")) or bool(args.threshold_sweep)
        th_list: list[float] | None = list(ts_cfg["thresholds"]) if sweep_on and ts_cfg.get("thresholds") else None

        dw_eff: Dict[str, Any] = {
            "enabled": True,
            "mode": mode,
            "weight_mode": weight_mode,
            "fov_min": fov_min_v,
            "tau": float(dw.get("tau", 0.7)),
            "alpha": float(dw.get("alpha", 1.0)),
            "gamma": gamma_cfg,
            "gamma_effective": gamma_effective,
            "weight_warmup_epochs": int(dw.get("warmup_epochs", 5)),
            "min_weight": float(dw.get("min_weight", 1.0)),
            "max_weight": float(dw.get("max_weight", 2.0)),
            "normalize_weights_in_batch": bool(dw.get("normalize_weights_in_batch", True)),
            "save_weight_stats": bool(log_cfg.get("save_weight_stats", True)),
            "shuffle_scores_in_batch": shuffle_ab,
            "threshold_sweep": {"enabled": sweep_on, "thresholds": th_list},
            "sampling": {
                "use_weighted_sampler": use_weighted_sampler,
                "hard_sampling_weight": float(sam_pre.get("hard_sampling_weight", 2.0)),
                "weighted_sampler_hard_mask": (
                    f"{target_col} > tau (no FOV gate)" if use_weighted_sampler else "n/a"
                ),
            },
            "debug_save_first_weighted_batch": bool(dw.get("debug_save_first_weighted_batch", False)),
        }
        probe_cfg = dw.get("probe") or {}
        if mode == "frozen_probe":
            ck = dw.get("probe_checkpoint")
            if not ck:
                raise ValueError("difficulty_weighting.probe_checkpoint required for mode frozen_probe")
            ck_path = Path(ck)
            if not ck_path.is_absolute():
                ck_path = root / ck_path
            if not ck_path.is_file():
                raise FileNotFoundError(f"probe_checkpoint not found: {ck_path}")
            dw_eff["probe_checkpoint"] = str(ck_path.resolve())
            dw_eff["probe"] = {
                "in_channels": int(probe_cfg.get("in_channels", 3)),
                "base_channels": int(probe_cfg.get("base_channels", 32)),
                "mlp_hidden": int(probe_cfg.get("mlp_hidden", 64)),
                "dropout": float(probe_cfg.get("dropout", 0.0)),
            }
        dw_eff["ramp_epochs"] = int(dw.get("ramp_epochs", 0))
        if mode == "joint":
            dw_eff["joint_calib_epochs"] = int(dw.get("joint_calib_epochs", 10))
            dw_eff["joint_oracle_weight_epochs"] = int(dw.get("joint_oracle_weight_epochs", 0))
            dw_eff["lambda_diff"] = float(dw.get("lambda_diff", 0.5))
            dw_eff["pred_weight_detach"] = bool(dw.get("pred_weight_detach", True))

        doc_oracle = _build_config_used(
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
        )
        doc_oracle["sampling_effective"] = dw_eff["sampling"]
        doc_oracle["threshold_sweep_effective"] = dw_eff["threshold_sweep"]
        _dump_config_used(out_dir, doc_oracle)

        dw_common = dict(
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            out_dir=out_dir,
            total_epochs=total_ep,
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

        if mode == "oracle":
            train_task3_baseline_oracle_weighted(
                model=UNetBaseline(in_channels=3, base=32).to(device),
                shuffle_scores_in_batch=shuffle_ab,
                weight_mode=weight_mode,
                fov_min=fov_min_v,
                threshold_sweep_thresholds=th_list,
                debug_save_first_weighted_batch=bool(dw.get("debug_save_first_weighted_batch", False)),
                **dw_common,
            )
            print(f"Done (oracle-weighted baseline). Artifacts in {out_dir}")
        elif mode == "frozen_probe":
            probe = load_frozen_sce_probe(
                ck_path,
                device,
                in_channels=int(probe_cfg.get("in_channels", 3)),
                base_channels=int(probe_cfg.get("base_channels", 32)),
                mlp_hidden=int(probe_cfg.get("mlp_hidden", 64)),
                dropout=float(probe_cfg.get("dropout", 0.0)),
            )
            train_task3_baseline_probe_weighted(
                model=UNetBaseline(in_channels=3, base=32).to(device),
                probe=probe,
                ramp_epochs=int(dw.get("ramp_epochs", 0)),
                threshold_sweep_thresholds=th_list,
                debug_save_first_weighted_batch=bool(dw.get("debug_save_first_weighted_batch", False)),
                **dw_common,
            )
            print(f"Done (frozen-probe-weighted baseline). Artifacts in {out_dir}")
        else:
            train_task3_joint_difficulty(
                model=UNetBaselineJointDifficulty(in_channels=3, base=32).to(device),
                joint_calib_epochs=int(dw.get("joint_calib_epochs", 10)),
                joint_oracle_weight_epochs=int(dw.get("joint_oracle_weight_epochs", 0)),
                lambda_diff=float(dw.get("lambda_diff", 0.5)),
                pred_weight_detach=bool(dw.get("pred_weight_detach", True)),
                ramp_epochs=int(dw.get("ramp_epochs", 0)),
                threshold_sweep_thresholds=th_list,
                debug_save_first_weighted_batch=bool(dw.get("debug_save_first_weighted_batch", False)),
                **dw_common,
            )
            print(f"Done (joint-difficulty baseline). Artifacts in {out_dir}")
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
    ts_cfg_b = cfg.get("threshold_sweep") or {}
    sweep_b = bool(ts_cfg_b.get("enabled")) or bool(args.threshold_sweep)
    th_list_b: list[float] | None = list(ts_cfg_b["thresholds"]) if sweep_b and ts_cfg_b.get("thresholds") else None
    doc_bl = _build_config_used(
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
    )
    doc_bl["threshold_sweep_effective"] = {"enabled": sweep_b, "thresholds": th_list_b}
    doc_bl["sampling_effective"] = {
        "use_weighted_sampler": use_weighted_sampler,
        "hard_sampling_weight": float(sam_pre.get("hard_sampling_weight", 2.0)),
        "note": "Weighted sampler only applies when difficulty_weighting.mode=oracle and enabled.",
    }
    _dump_config_used(out_dir, doc_bl)

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
        threshold_sweep_thresholds=th_list_b,
    )
    print(f"Done. Artifacts in {out_dir}")


if __name__ == "__main__":
    main()
