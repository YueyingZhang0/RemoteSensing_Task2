#!/usr/bin/env python3
"""External-style evaluation with dataset-agnostic thin-vessel hard proxy.

This script is **read-only** w.r.t. canonical runs: it never writes into existing canonical output_dirs.
All results must go to a NEW output directory.

Outputs in out_dir:
  - config_used.yaml
  - train_split_ids.json / val_split_ids.json / test_split_ids.json
  - val_metrics.json
  - hard_metrics_thinvessel.json
  - hard_mask_stats.json
  - cldice_summary.json
  - per_image_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics.cldice import cldice_score  # noqa: E402
from src.metrics.thinvessel_proxy import (  # noqa: E402
    ThinVesselProxyConfig,
    dice_on_region,
    hard_mask_stats,
    thin_proxy_score_from_masks,
    thin_vessel_hard_mask,
)
from src.external.hrf_paths import hrf_case_paths  # noqa: E402
from src.models.task3_unet import UNetBaseline, UNetBaselineJointDifficulty  # noqa: E402


def _read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _read_mask01(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    return (arr > 0.5).astype(np.uint8)


def _discover_drive_training(root: Path) -> List[Dict[str, Path]]:
    # This repo currently contains the DRIVE *training* set only (21..40).
    img_dir = root / "images"
    gt_dir = root / "1st_manual"
    fov_dir = root / "mask"
    cases: List[Dict[str, Path]] = []
    for i in range(21, 41):
        sid = f"{i:02d}"
        ip = img_dir / f"{sid}_training.tif"
        gt = gt_dir / f"{sid}_manual1.gif"
        fov = fov_dir / f"{sid}_training_mask.gif"
        if not ip.is_file():
            raise FileNotFoundError(ip)
        if not gt.is_file():
            raise FileNotFoundError(gt)
        if not fov.is_file():
            raise FileNotFoundError(fov)
        cases.append({"case_id": sid, "image": ip, "gt": gt, "fov": fov})
    return cases


@torch.no_grad()
def _infer_sliding_window_prob(
    model: torch.nn.Module,
    image_rgb01: np.ndarray,
    *,
    patch: int,
    stride: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    h, w, _ = image_rgb01.shape
    if patch > h or patch > w:
        pad_h = max(0, patch - h)
        pad_w = max(0, patch - w)
        image_rgb01 = np.pad(image_rgb01, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        h, w, _ = image_rgb01.shape

    ys = list(range(0, max(1, h - patch + 1), stride))
    xs = list(range(0, max(1, w - patch + 1), stride))
    if ys[-1] != h - patch:
        ys.append(h - patch)
    if xs[-1] != w - patch:
        xs.append(w - patch)

    acc = np.zeros((h, w), dtype=np.float32)
    cnt = np.zeros((h, w), dtype=np.float32)
    coords: List[Tuple[int, int]] = [(y, x) for y in ys for x in xs]

    model.eval()
    for i in range(0, len(coords), batch_size):
        batch_coords = coords[i : i + batch_size]
        patches = []
        for (y, x) in batch_coords:
            p = image_rgb01[y : y + patch, x : x + patch, :]
            patches.append(torch.from_numpy(p).permute(2, 0, 1))
        xb = torch.stack(patches, dim=0).to(device=device, dtype=torch.float32)
        out = model(xb)
        logits = out[0] if isinstance(out, tuple) else out
        prob = torch.sigmoid(logits).detach().cpu().numpy()  # (B,1,patch,patch)
        for j, (y, x) in enumerate(batch_coords):
            acc[y : y + patch, x : x + patch] += prob[j, 0]
            cnt[y : y + patch, x : x + patch] += 1.0

    return acc / np.maximum(cnt, 1.0)


def _dice_recall(pred: np.ndarray, gt: np.ndarray, fov: Optional[np.ndarray]) -> Tuple[float, float]:
    p = (pred > 0).astype(bool)
    g = (gt > 0).astype(bool)
    if fov is not None:
        m = (fov > 0).astype(bool)
        p = p & m
        g = g & m
    tp = float((p & g).sum())
    fp = float((p & ~g).sum())
    fn = float((~p & g).sum())
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-5)
    recall = tp / (tp + fn + 1e-5)
    return float(dice), float(recall)


def _load_model(kind: str, ckpt: Path, device: torch.device) -> torch.nn.Module:
    if kind == "baseline":
        m: torch.nn.Module = UNetBaseline(in_channels=3, base=32).to(device)
    elif kind == "joint":
        m = UNetBaselineJointDifficulty(in_channels=3, base=32).to(device)
    else:
        raise ValueError(f"Unknown kind={kind!r}")
    s = torch.load(ckpt, map_location=device, weights_only=False)
    m.load_state_dict(s["model_state_dict"])
    m.eval()
    return m


def _write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="External-style eval with thin-vessel hard proxy.")
    ap.add_argument(
        "--dataset",
        type=str,
        choices=("drive_training", "chase_db1_1st", "hrf_all"),
        required=True,
    )
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--kind", type=str, choices=("baseline", "joint"), required=True)
    ap.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Output directory. By default must be NEW; use --append_to_existing to add files into an existing run dir.",
    )
    ap.add_argument(
        "--append_to_existing",
        action="store_true",
        help="Allow writing into an existing out_dir, but refuse to overwrite any of the target filenames.",
    )
    ap.add_argument(
        "--name_prefix",
        type=str,
        default="",
        help="Optional prefix for all written metric files (useful when appending into a fine-tune directory).",
    )
    ap.add_argument("--split_ids_json", type=Path, default=None, help="Path to *_split_ids.json with {'ids':[...]} for evaluation set.")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=6)
    ap.add_argument("--r_th", type=int, default=2)
    ap.add_argument("--dilate", type=int, default=1)
    args = ap.parse_args()

    out_dir = (ROOT / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir.resolve()
    if out_dir.exists() and any(out_dir.iterdir()) and not args.append_to_existing:
        raise SystemExit(f"Refusing to overwrite non-empty out_dir: {out_dir} (use --append_to_existing)")
    out_dir.mkdir(parents=True, exist_ok=True)

    # In append mode, we never overwrite existing training artifacts (config_used, history, etc.).
    prefix = str(args.name_prefix or "")
    def pfx(name: str) -> str:
        return f"{prefix}{name}" if prefix else name

    will_write = [
        pfx("val_metrics.json"),
        pfx("hard_metrics_thinvessel.json"),
        pfx("hard_mask_stats.json"),
        pfx("cldice_summary.json"),
        pfx("per_image_metrics.csv"),
    ]
    if args.append_to_existing:
        for name in will_write:
            p = out_dir / name
            if p.exists():
                raise SystemExit(f"Refusing to overwrite existing file in append mode: {p}")

    split_ids: Optional[List[str]] = None
    if args.split_ids_json is not None:
        sp = args.split_ids_json if args.split_ids_json.is_absolute() else (ROOT / args.split_ids_json).resolve()
        obj = json.loads(sp.read_text(encoding="utf-8"))
        split_ids = [str(x) for x in obj.get("ids", [])]

    if args.dataset == "drive_training":
        cases = _discover_drive_training(ROOT)
        dataset_name = "DRIVE training (21-40; 1st_manual; FOV mask)"
        if split_ids is not None:
            wanted = set(split_ids)
            cases = [c for c in cases if c["case_id"] in wanted]
    elif args.dataset == "chase_db1_1st":
        chase_dir = ROOT / "CHASE_DB1"
        if not chase_dir.is_dir():
            raise FileNotFoundError(chase_dir)
        if split_ids is None:
            raise SystemExit("--split_ids_json is required for chase_db1_1st")
        cases = []
        for cid in split_ids:
            ip = chase_dir / f"{cid}.jpg"
            gt = chase_dir / f"{cid}_1stHO.png"
            if not ip.is_file():
                raise FileNotFoundError(ip)
            if not gt.is_file():
                raise FileNotFoundError(gt)
            cases.append({"case_id": cid, "image": ip, "gt": gt})
        dataset_name = "CHASE_DB1 (1st observer; no FOV)"
    elif args.dataset == "hrf_all":
        if split_ids is None:
            raise SystemExit("--split_ids_json is required for hrf_all")
        cases = [hrf_case_paths(cid, ROOT) for cid in split_ids]
        dataset_name = "HRF all (manual1; FOV mask)"
    else:
        raise SystemExit(f"Unsupported dataset {args.dataset!r}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(args.kind, args.ckpt, device)

    cfg = ThinVesselProxyConfig(r_th=int(args.r_th), dilate_iters=int(args.dilate))

    per_image: List[Dict[str, Any]] = []
    hard_stats_rows: List[Dict[str, Any]] = []

    for c in cases:
        img = _read_rgb(c["image"])
        gt = _read_mask01(c["gt"])
        fov = _read_mask01(c["fov"]) if c.get("fov") else None

        prob = _infer_sliding_window_prob(
            model, img, patch=int(args.patch), stride=int(args.stride), batch_size=int(args.batch_size), device=device
        )
        pred = (prob >= float(args.threshold)).astype(np.uint8)

        dice, recall = _dice_recall(pred, gt, fov)
        if fov is not None:
            pred_v = (pred > 0) & (fov > 0)
            gt_v = (gt > 0) & (fov > 0)
        else:
            pred_v = (pred > 0)
            gt_v = (gt > 0)
        cld = float(cldice_score(pred_v, gt_v))

        hard = thin_vessel_hard_mask(gt, fov_mask=fov, cfg=cfg)
        hard_dice = dice_on_region(pred, gt, region_mask=hard)
        score = thin_proxy_score_from_masks(vessel_mask=gt, hard_mask=hard)
        stats = hard_mask_stats(vessel_mask=gt, hard_mask=hard, fov_mask=fov)

        per_image.append(
            {
                "dataset": dataset_name,
                "case_id": c["case_id"],
                "dice@0.5": dice,
                "recall@0.5": recall,
                "cldice@0.5": cld,
                "hard_dice_thinvessel@0.5": hard_dice,
                "thin_proxy_score_gt": score,
                "image_path": str(c["image"].resolve()),
                "gt_path": str(c["gt"].resolve()),
                "fov_path": str(c["fov"].resolve()) if c.get("fov") else "",
            }
        )
        hard_stats_rows.append({"case_id": c["case_id"], **stats})

    # Aggregates
    d = np.array([float(r["dice@0.5"]) for r in per_image], dtype=np.float64)
    rcl = np.array([float(r["recall@0.5"]) for r in per_image], dtype=np.float64)
    cd = np.array([float(r["cldice@0.5"]) for r in per_image], dtype=np.float64)
    hd = np.array([float(r["hard_dice_thinvessel@0.5"]) for r in per_image], dtype=np.float64)

    val_metrics = {
        "dataset": dataset_name,
        "threshold": float(args.threshold),
        "val_dice_mean": float(np.mean(d)),
        "val_recall_mean": float(np.mean(rcl)),
        "cldice_mean": float(np.mean(cd)),
        "n_images": int(len(per_image)),
    }
    hard_metrics = {
        "dataset": dataset_name,
        "thin_vessel_proxy": {"r_th": int(args.r_th), "dilate_iters": int(args.dilate)},
        "threshold": float(args.threshold),
        "hard_dice_mean": float(np.mean(hd)),
        "n_images": int(len(per_image)),
        "note": "Hard Dice computed on dataset-agnostic thin-vessel hard proxy mask derived from GT.",
    }

    # Hard mask stats summary
    hard_pixel_ratio = [float(x["hard_pixel_ratio_in_fov"]) for x in hard_stats_rows if not np.isnan(float(x["hard_pixel_ratio_in_fov"]))]
    hard_vessel_ratio = [float(x["hard_vessel_ratio"]) for x in hard_stats_rows if not np.isnan(float(x["hard_vessel_ratio"]))]
    hard_mask_stats_json = {
        "dataset": dataset_name,
        "thin_vessel_proxy": {"r_th": int(args.r_th), "dilate_iters": int(args.dilate)},
        "per_image": hard_stats_rows,
        "summary": {
            "hard_pixel_ratio_in_fov_mean": float(np.mean(hard_pixel_ratio)) if hard_pixel_ratio else float("nan"),
            "hard_vessel_ratio_mean": float(np.mean(hard_vessel_ratio)) if hard_vessel_ratio else float("nan"),
        },
    }

    # clDice summary format (local)
    cldice_summary = {
        "threshold_fixed": float(args.threshold),
        "cldice_mean": float(np.mean(cd)),
        "n_images": int(len(per_image)),
        "note": "clDice is supportive; computed after applying FOV mask if available.",
    }

    # Split ids files (re-eval => N/A)
    # Split id files: only write them in non-append mode (dedicated ext_* eval dirs).
    if not args.append_to_existing:
        _write_json(out_dir / "train_split_ids.json", {"note": "N/A (eval only)", "ids": []})
        _write_json(out_dir / "val_split_ids.json", {"note": "N/A (eval only)", "ids": []})
        _write_json(
            out_dir / "test_split_ids.json",
            {"note": "Cases evaluated (from split_ids_json if provided)", "ids": [c["case_id"] for c in cases]},
        )

    # config_used.yaml
    cfg_used = {
        "evaluation_type": "external_style_thinvessel_eval",
        "dataset": args.dataset,
        "dataset_name": dataset_name,
        "split_ids_json": (str(args.split_ids_json) if args.split_ids_json is not None else None),
        "threshold": float(args.threshold),
        "sliding_window": {"patch": int(args.patch), "stride": int(args.stride), "batch_size": int(args.batch_size)},
        "thin_vessel_proxy": {"r_th": int(args.r_th), "dilate_iters": int(args.dilate)},
        "checkpoint": str(args.ckpt.resolve()),
        "model_kind": args.kind,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if not args.append_to_existing:
        (out_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg_used, sort_keys=False), encoding="utf-8")

    _write_json(out_dir / pfx("val_metrics.json"), val_metrics)
    _write_json(out_dir / pfx("hard_metrics_thinvessel.json"), hard_metrics)
    _write_json(out_dir / pfx("hard_mask_stats.json"), hard_mask_stats_json)
    _write_json(out_dir / pfx("cldice_summary.json"), cldice_summary)

    # per-image CSV
    csv_path = out_dir / pfx("per_image_metrics.csv")
    cols = list(per_image[0].keys()) if per_image else []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in per_image:
            w.writerow(r)

    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()

