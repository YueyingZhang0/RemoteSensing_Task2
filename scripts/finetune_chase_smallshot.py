#!/usr/bin/env python3
"""Small-shot fine-tune on CHASE_DB1 using thin-vessel proxy difficulty targets.

Runs:
  - P0 (baseline UNet)
  - CDC/J3 (joint difficulty UNet), preserving calibration-first + coupled weighting schedule

All results are written to NEW directories under outputs/ft_chase_* and never overwrite canonical runs.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.external_thinvessel_crop_dataset import (  # noqa: E402
    ExternalCropConfig,
    ExternalThinVesselCropDataset,
)
from src.external.hrf_paths import hrf_case_paths  # noqa: E402
from src.models.task3_unet import UNetBaseline, UNetBaselineJointDifficulty  # noqa: E402
from src.training.task3_engine import collate_seg_sce, train_task3, train_task3_joint_difficulty  # noqa: E402


def _load_ids(p: Path) -> List[str]:
    obj = json.loads(p.read_text(encoding="utf-8"))
    return [str(x) for x in obj.get("ids", [])]


def _cases_from_ids(dataset: str, ids: List[str]) -> List[Dict[str, Path]]:
    if dataset == "chase":
        chase_dir = ROOT / "CHASE_DB1"
        cases = []
        for cid in ids:
            ip = chase_dir / f"{cid}.jpg"
            gt = chase_dir / f"{cid}_1stHO.png"
            if not ip.is_file():
                raise FileNotFoundError(ip)
            if not gt.is_file():
                raise FileNotFoundError(gt)
            cases.append({"case_id": cid, "image": ip, "gt": gt})
        return cases
    if dataset == "hrf":
        return [hrf_case_paths(cid, ROOT) for cid in ids]
    raise ValueError(f"Unknown dataset={dataset!r}")


def _dump_config_used(out_dir: Path, doc: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_used.yaml").write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")


def _set_reproducibility(seed: int) -> torch.Generator:
    """Align PyTorch / NumPy / Python RNG and DataLoader shuffles with ``--seed`` for multi-seed runs."""
    s = int(seed)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    g = torch.Generator()
    g.manual_seed(s)
    return g


def main() -> None:
    ap = argparse.ArgumentParser(description="Small-shot fine-tune on CHASE_DB1 or HRF/all from canonical checkpoints.")
    ap.add_argument("--dataset", type=str, choices=("chase", "hrf"), default="chase")
    ap.add_argument("--split_dir", type=Path, required=True, help="Directory containing train/val/test_split_ids.json")
    ap.add_argument("--method", type=str, choices=("p0", "j3"), required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--image_size", type=int, default=128)
    ap.add_argument("--crops_per_image", type=int, default=64)
    ap.add_argument("--out_dir", type=Path, required=True)
    args = ap.parse_args()

    split_dir = args.split_dir if args.split_dir.is_absolute() else (ROOT / args.split_dir).resolve()
    train_ids = _load_ids(split_dir / "train_split_ids.json")
    val_ids = _load_ids(split_dir / "val_split_ids.json")
    test_ids = _load_ids(split_dir / "test_split_ids.json")

    out_dir = args.out_dir if args.out_dir.is_absolute() else (ROOT / args.out_dir).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty out_dir: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Persist split ids into the run directory (required)
    (out_dir / "train_split_ids.json").write_text(json.dumps({"ids": train_ids}, indent=2), encoding="utf-8")
    (out_dir / "val_split_ids.json").write_text(json.dumps({"ids": val_ids}, indent=2), encoding="utf-8")
    (out_dir / "test_split_ids.json").write_text(json.dumps({"ids": test_ids}, indent=2), encoding="utf-8")

    train_cases = _cases_from_ids(args.dataset, train_ids)
    val_cases = _cases_from_ids(args.dataset, val_ids)

    crop_cfg = ExternalCropConfig(image_size=int(args.image_size), crops_per_image=int(args.crops_per_image))
    train_ds = ExternalThinVesselCropDataset(cases=train_cases, cfg=crop_cfg, seed=int(args.seed), augment=True)
    val_ds = ExternalThinVesselCropDataset(cases=val_cases, cfg=crop_cfg, seed=int(args.seed) + 10_000, augment=False)

    dl_gen = _set_reproducibility(int(args.seed))

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        generator=dl_gen,
        num_workers=0,
        collate_fn=collate_seg_sce,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_seg_sce,
        pin_memory=pin,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.method == "p0":
        ckpt = ROOT / "outputs" / "task3_v2_1_baseline_same_split" / "best_model.pt"
        model = UNetBaseline(in_channels=3, base=32).to(device)
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])

        _dump_config_used(
            out_dir,
            {
                "run_type": "fine_tune_smallshot",
                "dataset": "CHASE_DB1" if args.dataset == "chase" else "HRF/all",
                "gt": "1st observer" if args.dataset == "chase" else "manual1 + FOV",
                "method": "P0 baseline",
                "init_checkpoint": str(ckpt.resolve()),
                "seed": int(args.seed),
                "epochs": int(args.epochs),
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "batch_size": int(args.batch_size),
                "image_size": int(args.image_size),
                "crops_per_image": int(args.crops_per_image),
                "thin_vessel_proxy": {"r_th": 2, "dilate_iters": 1, "score": "hard_vessel_ratio_in_crop"},
                "reproducibility": {
                    "torch_manual_seed": int(args.seed),
                    "numpy_seed": int(args.seed),
                    "python_random_seed": int(args.seed),
                    "train_dataloader_generator_seed": int(args.seed),
                    "external_crop_dataset_train_seed": int(args.seed),
                    "external_crop_dataset_val_seed": int(args.seed) + 10_000,
                    "cudnn_deterministic": True,
                    "cudnn_benchmark": False,
                    "note": "Test metrics after fine-tune: external_test_* JSON (test split, same protocol as ext_*_zeroshot).",
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

        # Baseline fine-tune: segmentation-only training. We map epochs -> warmup_epochs, no joint phase.
        train_task3(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            ours=False,
            out_dir=out_dir,
            warmup_epochs=int(args.epochs),
            joint_epochs=0,
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            lambda_sce=0.0,
            threshold_sweep_thresholds=[0.5],  # external module uses fixed tau=0.5
        )
        return

    # J3 / CDC joint difficulty fine-tune (preserve mechanism)
    ckpt = ROOT / "outputs" / "task3_J3_joint_detach_false" / "best_model.pt"
    model = UNetBaselineJointDifficulty(in_channels=3, base=32).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])

    # Use the same *definitions* as canonical J3, but keep total epochs small for small-shot.
    tau = 0.6
    alpha = 0.5
    gamma = 2.0
    difficulty_warmup_epochs = 5
    joint_calib_epochs = min(15, max(5, int(args.epochs // 2)))
    joint_oracle_weight_epochs = 0
    lambda_diff = 0.5
    pred_weight_detach = False
    ramp_epochs = min(10, max(0, int(args.epochs // 3)))

    _dump_config_used(
        out_dir,
        {
            "run_type": "fine_tune_smallshot",
            "dataset": "CHASE_DB1" if args.dataset == "chase" else "HRF/all",
            "gt": "1st observer" if args.dataset == "chase" else "manual1 + FOV",
            "method": "CDC (proposed; internal J3)",
            "init_checkpoint": str(ckpt.resolve()),
            "seed": int(args.seed),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "batch_size": int(args.batch_size),
            "image_size": int(args.image_size),
            "crops_per_image": int(args.crops_per_image),
            "thin_vessel_proxy": {"r_th": 2, "dilate_iters": 1, "score": "hard_vessel_ratio_in_crop"},
            "difficulty_weighting": {
                "mode": "joint",
                "tau": tau,
                "alpha": alpha,
                "gamma": gamma,
                "weight_warmup_epochs": difficulty_warmup_epochs,
                "joint_calib_epochs": joint_calib_epochs,
                "joint_oracle_weight_epochs": joint_oracle_weight_epochs,
                "lambda_diff": lambda_diff,
                "pred_weight_detach": pred_weight_detach,
                "ramp_epochs": ramp_epochs,
                "min_weight": 1.0,
                "max_weight": 1.5,
                "normalize_weights_in_batch": True,
            },
            "reproducibility": {
                "torch_manual_seed": int(args.seed),
                "numpy_seed": int(args.seed),
                "python_random_seed": int(args.seed),
                "train_dataloader_generator_seed": int(args.seed),
                "external_crop_dataset_train_seed": int(args.seed),
                "external_crop_dataset_val_seed": int(args.seed) + 10_000,
                "cudnn_deterministic": True,
                "cudnn_benchmark": False,
                "note": "Test metrics after fine-tune: external_test_* JSON (test split, same protocol as ext_*_zeroshot).",
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )

    train_task3_joint_difficulty(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        out_dir=out_dir,
        total_epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        tau=tau,
        alpha=alpha,
        gamma=gamma,
        difficulty_warmup_epochs=int(difficulty_warmup_epochs),
        min_weight=1.0,
        max_weight=1.5,
        normalize_weights_in_batch=True,
        save_weight_stats=True,
        joint_calib_epochs=int(joint_calib_epochs),
        joint_oracle_weight_epochs=int(joint_oracle_weight_epochs),
        lambda_diff=float(lambda_diff),
        pred_weight_detach=bool(pred_weight_detach),
        ramp_epochs=int(ramp_epochs),
        threshold_sweep_thresholds=[0.5],
        debug_save_first_weighted_batch=True,
    )


if __name__ == "__main__":
    main()

