#!/usr/bin/env python3
"""Evaluate nnU-Net v2 CHASE predictions with the **same** thin-vessel / clDice / Dice@0.5 protocol as
``eval_external_thinvessel.py``, after resizing each probability map to GT resolution.

**Primary metric (CHASE zero-shot matrix):** mean Hard Dice @0.5 (thin-vessel proxy).
Dice / Recall / clDice @0.5 are secondary.

nnU-Net uses its **standard** internal preprocessing / resampling / normalization; this is declared in
written JSON notes and should be repeated in the manuscript table footnote.

Inputs:
  --pred-dir: folder with ``{case_id}.npz`` (``probabilities`` key) and/or ``{case_id}.png`` (same as Task3 eval).

Does not write into canonical Task3 nnU-Net eval dirs by default.

Example:
  python scripts/eval_nnunet_chase_thinvessel.py \\
    --pred-dir nnUNet_predictions/chase_zeroshot_prob \\
    --split_ids_json outputs/ext_chase_split_seed42/test_split_ids.json \\
    --out_dir outputs/ext_chase_nnunet_v2_zeroshot
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
import yaml
from PIL import Image
from scipy.ndimage import zoom

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics.cldice import cldice_score  # noqa: E402
from src.metrics.thinvessel_proxy import (  # noqa: E402
    ThinVesselProxyConfig,
    dice_on_region,
    hard_mask_stats,
    thin_proxy_score_from_masks,
    thin_vessel_hard_mask,
)


def _read_mask01(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    return (arr > 0.5).astype(np.uint8)


def _resize_to(prob: np.ndarray, shape_hw: Tuple[int, int], order: int = 1) -> np.ndarray:
    th, tw = shape_hw
    sh, sw = prob.shape
    if (sh, sw) == (th, tw):
        return prob.astype(np.float32)
    zy, zx = th / sh, tw / sw
    return zoom(prob.astype(np.float64), (zy, zx), order=order).astype(np.float32)


def _load_nnunet_pred(pred_dir: Path, case_id: str, gt_hw: Tuple[int, int], file_ending: str) -> Tuple[np.ndarray, str]:
    npz_path = pred_dir / f"{case_id}.npz"
    png_path = pred_dir / f"{case_id}{file_ending}"
    if npz_path.is_file():
        z = np.load(npz_path)
        prob_c = z["probabilities"]
        if prob_c.ndim == 4 and prob_c.shape[1] == 1:
            prob_c = prob_c[:, 0]
        if prob_c.ndim != 3:
            raise ValueError(f"Unexpected probabilities shape for {case_id}: {prob_c.shape}")
        if prob_c.shape[0] >= 2:
            p = prob_c[1].astype(np.float32)
        else:
            p = prob_c[0].astype(np.float32)
        p = np.clip(p, 0.0, 1.0)
        p = _resize_to(p, gt_hw, order=1)
        return p, "prob"
    if png_path.is_file():
        seg = np.asarray(Image.open(png_path).convert("L"), dtype=np.uint8)
        seg = _resize_to(seg.astype(np.float32), gt_hw, order=0)
        p = (seg > 0).astype(np.float32)
        return p, "seg"
    raise FileNotFoundError(f"No prediction for case_id={case_id} in {pred_dir} (.npz or {file_ending})")


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


def _write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="CHASE nnU-Net zero-shot eval (thin-vessel hard Dice primary).")
    ap.add_argument("--pred-dir", type=Path, required=True)
    ap.add_argument("--split_ids_json", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument(
        "--chase_dir",
        type=Path,
        default=None,
        help="CHASE_DB1 root (default: <repo>/CHASE_DB1)",
    )
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--r_th", type=int, default=2)
    ap.add_argument("--dilate", type=int, default=1)
    ap.add_argument("--file_ending", type=str, default=".png")
    ap.add_argument("--max_images", type=int, default=0, help="If >0, only first N split ids.")
    args = ap.parse_args()

    chase = args.chase_dir if args.chase_dir is not None else ROOT / "CHASE_DB1"
    chase = chase.resolve() if not chase.is_absolute() else chase
    pred_dir = args.pred_dir if args.pred_dir.is_absolute() else (ROOT / args.pred_dir).resolve()
    sp = args.split_ids_json if args.split_ids_json.is_absolute() else (ROOT / args.split_ids_json).resolve()
    out_dir = args.out_dir if args.out_dir.is_absolute() else (ROOT / args.out_dir).resolve()

    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"Refusing non-empty out_dir: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = [str(x) for x in json.loads(sp.read_text(encoding="utf-8")).get("ids", [])]
    if int(args.max_images) > 0:
        ids = ids[: int(args.max_images)]

    dataset_name = "CHASE_DB1 (1st observer; no FOV); nnU-Net v2 standard inference pipeline"
    cfg = ThinVesselProxyConfig(r_th=int(args.r_th), dilate_iters=int(args.dilate))

    per_image: List[Dict[str, Any]] = []
    hard_stats_rows: List[Dict[str, Any]] = []

    for cid in ids:
        ip = chase / f"{cid}.jpg"
        gt_path = chase / f"{cid}_1stHO.png"
        if not ip.is_file():
            raise FileNotFoundError(ip)
        if not gt_path.is_file():
            raise FileNotFoundError(gt_path)
        gt = _read_mask01(gt_path)
        gh, gw = gt.shape[:2]
        prob, mode = _load_nnunet_pred(pred_dir, cid, (gh, gw), args.file_ending)
        if prob.shape != (gh, gw):
            raise RuntimeError(f"prob shape {prob.shape} != GT {(gh, gw)} for {cid}")
        pred = (prob >= float(args.threshold)).astype(np.uint8)
        fov: Optional[np.ndarray] = None

        dice, recall = _dice_recall(pred, gt, fov)
        pred_v = pred > 0
        gt_v = gt > 0
        cld = float(cldice_score(pred_v, gt_v))

        hard = thin_vessel_hard_mask(gt, fov_mask=fov, cfg=cfg)
        hard_dice = dice_on_region(pred, gt, region_mask=hard)
        score = thin_proxy_score_from_masks(vessel_mask=gt, hard_mask=hard)
        stats = hard_mask_stats(vessel_mask=gt, hard_mask=hard, fov_mask=fov)

        per_image.append(
            {
                "dataset": dataset_name,
                "case_id": cid,
                "dice@0.5": dice,
                "recall@0.5": recall,
                "cldice@0.5": cld,
                "hard_dice_thinvessel@0.5": hard_dice,
                "thin_proxy_score_gt": score,
                "nnunet_pred_mode": mode,
                "image_path": str(ip.resolve()),
                "gt_path": str(gt_path.resolve()),
                "fov_path": "",
            }
        )
        hard_stats_rows.append({"case_id": cid, **stats})

    d = np.array([float(r["dice@0.5"]) for r in per_image], dtype=np.float64)
    rcl = np.array([float(r["recall@0.5"]) for r in per_image], dtype=np.float64)
    cd = np.array([float(r["cldice@0.5"]) for r in per_image], dtype=np.float64)
    hd = np.array([float(r["hard_dice_thinvessel@0.5"]) for r in per_image], dtype=np.float64)

    nnunet_disclaimer = (
        "nnU-Net v2 inference used its standard preprocessing, resampling, and normalization "
        "(not identical to the raw-image sliding-window PyTorch protocol used for P0/CDC/Attention/Swin). "
        "Probability maps were resized to CHASE GT resolution before thresholding at 0.5."
    )

    val_metrics = {
        "dataset": dataset_name,
        "threshold": float(args.threshold),
        "primary_metric": "hard_dice_thinvessel@0.5_mean",
        "secondary_metrics": ["val_dice_mean", "val_recall_mean", "cldice_mean"],
        "val_dice_mean": float(np.mean(d)),
        "val_recall_mean": float(np.mean(rcl)),
        "cldice_mean": float(np.mean(cd)),
        "n_images": int(len(per_image)),
        "inference_protocol": "nnunet_v2_standard",
        "inference_protocol_note": nnunet_disclaimer,
    }
    hard_metrics = {
        "dataset": dataset_name,
        "thin_vessel_proxy": {"r_th": int(args.r_th), "dilate_iters": int(args.dilate)},
        "threshold": float(args.threshold),
        "primary_metric": "hard_dice_mean",
        "hard_dice_mean": float(np.mean(hd)),
        "n_images": int(len(per_image)),
        "note": nnunet_disclaimer,
    }
    hard_pixel_ratio = [
        float(x["hard_pixel_ratio_in_fov"])
        for x in hard_stats_rows
        if not np.isnan(float(x["hard_pixel_ratio_in_fov"]))
    ]
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
    cldice_summary = {
        "threshold_fixed": float(args.threshold),
        "primary_metric": "hard_dice_thinvessel@0.5 (see hard_metrics_thinvessel.json)",
        "cldice_mean": float(np.mean(cd)),
        "n_images": int(len(per_image)),
        "note": "clDice is secondary (connectivity).",
    }

    _write_json(out_dir / "train_split_ids.json", {"note": "N/A (eval only)", "ids": []})
    _write_json(out_dir / "val_split_ids.json", {"note": "N/A (eval only)", "ids": []})
    _write_json(out_dir / "test_split_ids.json", {"note": "CHASE cases evaluated", "ids": ids})

    cfg_used = {
        "evaluation_type": "nnunet_chase_thinvessel_eval",
        "pred_dir": str(pred_dir),
        "split_ids_json": str(sp),
        "max_images": int(args.max_images),
        "primary_metric_chase_matrix": "Hard Dice @0.5 (thin-vessel proxy mean over evaluated images)",
        "secondary_metrics_chase_matrix": ["Dice @0.5", "Recall @0.5", "clDice @0.5"],
        "nnunet_protocol_disclaimer": nnunet_disclaimer,
        "threshold": float(args.threshold),
        "thin_vessel_proxy": {"r_th": int(args.r_th), "dilate_iters": int(args.dilate)},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg_used, sort_keys=False), encoding="utf-8")

    _write_json(out_dir / "val_metrics.json", val_metrics)
    _write_json(out_dir / "hard_metrics_thinvessel.json", hard_metrics)
    _write_json(out_dir / "hard_mask_stats.json", hard_mask_stats_json)
    _write_json(out_dir / "cldice_summary.json", cldice_summary)

    csv_path = out_dir / "per_image_metrics.csv"
    cols = list(per_image[0].keys()) if per_image else []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in per_image:
            w.writerow(r)

    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
