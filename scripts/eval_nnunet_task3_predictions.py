#!/usr/bin/env python3
"""Evaluate nnU-Net v2 predictions on Task3 val patches (same protocol as train_task3).

Reads GT masks from patch_mask_dir; predictions from nnUNet output folder.

**Preferred:** nnUNet saved **probabilities** (`--save_probabilities` -> `{case_id}.npz` with key `probabilities`,
shape [C,H,W], use foreground channel = 1 for vessel). Enables full threshold sweep.

**Fallback:** segmentation PNG only (`{case_id}.png`) -> Hard/val Dice only meaningful at 0.5; sweep is approximate.

Writes to --out-dir (default outputs/task3_nnunet_eval):
  val_metrics.json, hard_patch_metrics.json, threshold_metrics.json, nnunet_cldice.json

  python scripts/eval_nnunet_task3_predictions.py \\
    --pred-dir path/to/nnunet_predictions \\
    --config configs/task3_v2_1_baseline.yaml

Re-run export if you need case_id_map.json: python scripts/export_task3_for_nnunet_v2.py ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import zoom

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.main.train_task3 import _apply_hard_labels
from src.metrics.cldice import cldice_score
from src.training.nnunet_task3_case_ids import build_nnunet_case_rows
from src.training.splits import load_or_create_group_split, make_group_train_val_split
from src.utils.io import load_yaml, save_json


def _resize_to(prob: np.ndarray, shape_hw: Tuple[int, int], order: int = 1) -> np.ndarray:
    """Resize 2D array to (H,W) = shape_hw."""
    th, tw = shape_hw
    sh, sw = prob.shape
    if (sh, sw) == (th, tw):
        return prob.astype(np.float32)
    zy, zx = th / sh, tw / sw
    return zoom(prob.astype(np.float64), (zy, zx), order=order).astype(np.float32)


def _load_nnunet_pred(
    pred_dir: Path,
    case_id: str,
    gt_hw: Tuple[int, int],
    file_ending: str,
) -> Tuple[np.ndarray, str]:
    """Return (prob_hw float32 in [0,1], mode 'prob'|'seg')."""
    npz_path = pred_dir / f"{case_id}.npz"
    png_path = pred_dir / f"{case_id}{file_ending}"
    if npz_path.is_file():
        z = np.load(npz_path)
        prob_c = z["probabilities"]
        # nnU-Net may write probabilities as (C,H,W) or (C,1,H,W) for 2D.
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


def _dice_recall_one(pred_bin: np.ndarray, gt_bin: np.ndarray, smooth: float = 1e-5) -> Tuple[float, float]:
    p = pred_bin.astype(np.float32)
    g = gt_bin.astype(np.float32)
    inter = (p * g).sum()
    d = float((2 * inter + smooth) / (p.sum() + g.sum() + smooth))
    tp = ((p > 0) & (g > 0)).sum()
    fn = ((p == 0) & (g > 0)).sum()
    r = float(tp / (tp + fn + smooth))
    return d, r


def _sweep_thresholds(
    probs: np.ndarray,
    gts: np.ndarray,
    hard_mask: np.ndarray,
    thresholds: List[float],
) -> Dict[str, Any]:
    """probs, gts: (N,H,W); hard_mask (N,) bool."""
    per_t: Dict[str, Dict[str, float]] = {}
    hard_dice_by_t: List[Tuple[float, float]] = []
    val_dice_by_t: List[Tuple[float, float]] = []
    n = probs.shape[0]
    for t in thresholds:
        dices: List[float] = []
        recs: List[float] = []
        hdices: List[float] = []
        hrecs: List[float] = []
        for i in range(n):
            pred = probs[i] > float(t)
            g = gts[i] > 0.5
            di, ri = _dice_recall_one(pred, g)
            dices.append(di)
            recs.append(ri)
            if hard_mask[i]:
                hdices.append(di)
                hrecs.append(ri)
        val_dice = float(np.mean(dices)) if dices else float("nan")
        val_rec = float(np.mean(recs)) if recs else float("nan")
        if hdices:
            hd = float(np.mean(hdices))
            hr = float(np.mean(hrecs))
        else:
            hd = float("nan")
            hr = float("nan")
        key = f"{t:.2f}"
        per_t[key] = {
            "val_dice_mean": val_dice,
            "val_recall_mean": val_rec,
            "hard_dice_mean": hd,
            "hard_recall_mean": hr,
        }
        if not np.isnan(val_dice):
            val_dice_by_t.append((float(t), val_dice))
        if not np.isnan(hd):
            hard_dice_by_t.append((float(t), hd))
    best_hard_t = max(hard_dice_by_t, key=lambda x: x[1])[0] if hard_dice_by_t else None
    best_val_t = max(val_dice_by_t, key=lambda x: x[1])[0] if val_dice_by_t else None
    return {
        "thresholds": thresholds,
        "per_threshold": per_t,
        "best_hard_dice_threshold": best_hard_t,
        "best_val_dice_threshold": best_val_t,
        "fixed_threshold_reporting": 0.5,
        "note": "nnU-Net external predictions; prob from .npz if present else binary .png.",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Eval nnU-Net predictions vs Task3 val/hard protocol.")
    ap.add_argument("--pred-dir", type=str, required=True, help="Folder with {case_id}.npz and/or {case_id}.png")
    ap.add_argument("--config", type=str, default="configs/task3_v2_1_baseline.yaml")
    ap.add_argument("--out-dir", type=str, default="outputs/task3_nnunet_eval")
    ap.add_argument("--dataset-id", type=int, default=503)
    ap.add_argument(
        "--case-map",
        type=str,
        default=None,
        help="Optional case_id_map.json (default: nnUNet_raw/DatasetXXX_TASK3/case_id_map.json)",
    )
    args = ap.parse_args()

    root = _ROOT
    cfg = load_yaml(root / args.config)
    pred_dir = Path(args.pred_dir)
    if not pred_dir.is_dir():
        raise FileNotFoundError(pred_dir)

    meta = Path(cfg["metadata_csv"])
    if not meta.is_absolute():
        meta = root / meta
    msk_dir = Path(cfg["patch_mask_dir"])
    if not msk_dir.is_absolute():
        msk_dir = root / msk_dir

    target_col = cfg.get("target_col", "sci_res2_norm")
    group_col = cfg.get("group_col", "sample_id")
    split_json = cfg.get("split_json")
    df = pd.read_csv(meta)
    if split_json:
        df_tr, df_va = load_or_create_group_split(
            df, group_col, float(cfg.get("val_ratio", 0.2)),
            int(cfg.get("random_state", 42)), split_json, root,
        )
    else:
        df_tr, df_va = make_group_train_val_split(
            df, group_col, float(cfg.get("val_ratio", 0.2)), int(cfg.get("random_state", 42)),
        )
    df_tr, df_va = _apply_hard_labels(df_tr, df_va, target_col, float(cfg.get("hard_ratio", 0.2)))

    ds_name = f"Dataset{int(args.dataset_id):03d}_TASK3"
    cmap_path = Path(args.case_map) if args.case_map else root / "nnUNet_raw" / ds_name / "case_id_map.json"
    if cmap_path.is_file():
        case_rows = json.loads(cmap_path.read_text(encoding="utf-8"))
    else:
        case_rows = build_nnunet_case_rows(df_tr, df_va)

    p2c = {r["patch_name"]: r["case_id"] for r in case_rows}

    file_ending = ".png"
    dj = root / "nnUNet_raw" / ds_name / "dataset.json"
    if dj.is_file():
        file_ending = str(json.loads(dj.read_text(encoding="utf-8")).get("file_ending", ".png"))

    probs_list: List[np.ndarray] = []
    gts_list: List[np.ndarray] = []
    hard_flags: List[bool] = []
    modes: List[str] = []
    th05 = 0.5

    for _, row in df_va.iterrows():
        pname = str(row["patch_name"])
        cid = p2c[pname]
        mp = msk_dir / pname
        if not mp.is_file():
            raise FileNotFoundError(mp)
        gt = (np.asarray(Image.open(mp).convert("L"), dtype=np.float32) > 127).astype(np.float32)
        gh, gw = gt.shape
        prob, mode = _load_nnunet_pred(pred_dir, cid, (gh, gw), file_ending)
        probs_list.append(prob)
        gts_list.append(gt)
        hard_flags.append(bool(row.get("is_hard", False)))
        modes.append(mode)

    probs = np.stack(probs_list, axis=0)
    gts = np.stack(gts_list, axis=0)
    hard_arr = np.asarray(hard_flags, dtype=bool)
    used_seg_only = all(m == "seg" for m in modes)

    # val / hard at 0.5
    vd, vr = [], []
    hd, hr = [], []
    cld_g: List[float] = []
    cld_h: List[float] = []
    for i in range(probs.shape[0]):
        pred = probs[i] > th05
        g = gts[i] > 0.5
        di, ri = _dice_recall_one(pred, g)
        vd.append(di)
        vr.append(ri)
        if hard_arr[i]:
            hd.append(di)
            hr.append(ri)
            cld_h.append(cldice_score(pred.astype(bool), g.astype(bool)))
        cld_g.append(cldice_score(pred.astype(bool), g.astype(bool)))

    final_val = {
        "val_dice_mean": float(np.mean(vd)),
        "val_recall_mean": float(np.mean(vr)),
        "hard_dice_mean": float(np.mean(hd)) if hd else float("nan"),
        "hard_recall_mean": float(np.mean(hr)) if hr else float("nan"),
        "val_sci_corr": None,
    }
    hard_json = {
        "hard_dice_mean": final_val["hard_dice_mean"],
        "hard_recall_mean": final_val["hard_recall_mean"],
        "n_hard_patches_note": "subset of val where is_hard=True (top sci_res2_norm)",
        "source": "nnunet_external",
    }

    ts_cfg = cfg.get("threshold_sweep") or {}
    thresholds = list(ts_cfg.get("thresholds", [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]))
    tm = _sweep_thresholds(probs, gts, hard_arr, thresholds)
    if used_seg_only:
        tm["note"] = (
            tm.get("note", "")
            + " WARNING: only binary PNG predictions found; probability sweep is degenerate (0/1 probs)."
        )

    cldice_block = {
        "threshold": th05,
        "cldice_global": float(np.mean(cld_g)) if cld_g else float("nan"),
        "cldice_hard": float(np.mean(cld_h)) if cld_h else float("nan"),
        "n_val": len(cld_g),
        "n_hard": int(hard_arr.sum()),
        "note": "Topology-oriented supplement; same definition as eval_cldice_task3.py.",
    }

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(final_val, out_dir / "val_metrics.json")
    save_json(hard_json, out_dir / "hard_patch_metrics.json")
    save_json(tm, out_dir / "threshold_metrics.json")
    save_json(
        {
            "method": "nnunet_v2",
            "pred_dir": str(pred_dir.resolve()),
            "used_seg_png_only": used_seg_only,
            "cldice": cldice_block,
        },
        out_dir / "nnunet_cldice.json",
    )

    print("Wrote", out_dir)
    print(f"  val_dice@0.5={final_val['val_dice_mean']:.6f}  hard_dice@0.5={final_val['hard_dice_mean']:.6f}")
    print(f"  best_hard_dice_threshold={tm.get('best_hard_dice_threshold')}")
    print(f"  clDice global@0.5={cldice_block['cldice_global']:.6f}  hard@0.5={cldice_block['cldice_hard']:.6f}")
    if used_seg_only:
        print("  NOTE: use nnUNetv2_predict --save_probabilities for a real threshold sweep.")


if __name__ == "__main__":
    main()
