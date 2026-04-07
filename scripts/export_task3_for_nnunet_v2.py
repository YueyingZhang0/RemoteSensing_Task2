#!/usr/bin/env python3
"""Export Task 3 patches to nnU-Net v2 raw dataset layout (2D, RGB as 3 channel files).

Produces:
  {out}/Dataset{dataset_id:03d}_TASK3/
    dataset.json
    splits_final.json
    imagesTr/{case}_0000.png ... _0002.png
    labelsTr/{case}.png

Case IDs are derived from patch_name stems (sanitized). Train/val lists match split_json / group split.

After export, run nnU-Net v2 plan/train externally (see docs/nnunet_v2_task3_baseline.md).

  python scripts/export_task3_for_nnunet_v2.py --config configs/task3_v2_1_baseline.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from PIL import Image

from src.training.nnunet_task3_case_ids import build_nnunet_case_rows
from src.training.splits import load_or_create_group_split, make_group_train_val_split, mark_hard_patches
from src.utils.io import load_yaml


def _apply_hard_labels(df_tr: pd.DataFrame, df_va: pd.DataFrame, target_col: str, hard_ratio: float):
    df_va = mark_hard_patches(df_va, target_col, hard_ratio)
    thr = float(np.quantile(df_va[target_col].values, 1.0 - hard_ratio))
    df_tr = df_tr.copy()
    df_tr["is_hard"] = df_tr[target_col] >= thr
    return df_tr, df_va


def main() -> None:
    p = argparse.ArgumentParser(description="Export Task3 patches for nnU-Net v2.")
    p.add_argument("--config", type=str, default="configs/task3_v2_1_baseline.yaml")
    p.add_argument("--dataset_id", type=int, default=503, help="nnU-Net dataset id (e.g. 503 -> Dataset503_TASK3)")
    p.add_argument(
        "--out",
        type=str,
        default="nnUNet_raw",
        help="Output root (relative to project root unless absolute).",
    )
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg_path = Path(args.config)
    cfg = load_yaml(cfg_path) if cfg_path.is_file() else {}

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

    ds_name = f"Dataset{int(args.dataset_id):03d}_TASK3"
    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = root / out_root
    ds_dir = out_root / ds_name
    im_tr = ds_dir / "imagesTr"
    lb_tr = ds_dir / "labelsTr"
    im_tr.mkdir(parents=True, exist_ok=True)
    lb_tr.mkdir(parents=True, exist_ok=True)

    case_rows = build_nnunet_case_rows(df_tr, df_va)
    tr_by = df_tr.set_index("patch_name")
    va_by = df_va.set_index("patch_name")

    train_ids: list[str] = []
    val_ids: list[str] = []

    def write_case_with_id(cid: str, split: str, row: pd.Series) -> None:
        name = str(row["patch_name"])
        ip = img_dir / name
        mp = msk_dir / name
        if not ip.is_file() or not mp.is_file():
            raise FileNotFoundError(f"Missing image/mask for {name}")
        rgb = np.asarray(Image.open(ip).convert("RGB"), dtype=np.uint8)
        msk = np.asarray(Image.open(mp).convert("L"), dtype=np.uint8)
        msk_bin = (msk > 127).astype(np.uint8)

        for ch in range(3):
            ch_img = Image.fromarray(rgb[:, :, ch], mode="L")
            ch_img.save(im_tr / f"{cid}_{ch:04d}.png")
        Image.fromarray(msk_bin, mode="L").save(lb_tr / f"{cid}.png")
        if split == "train":
            train_ids.append(cid)
        else:
            val_ids.append(cid)

    for rec in case_rows:
        lo = tr_by if rec["split"] == "train" else va_by
        row = lo.loc[rec["patch_name"]]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        write_case_with_id(rec["case_id"], rec["split"], row)

    with open(ds_dir / "case_id_map.json", "w", encoding="utf-8") as f:
        json.dump(case_rows, f, indent=2)

    n_train = len(df_tr) + len(df_va)
    dataset_json = {
        "channel_names": {"0": "R", "1": "G", "2": "B"},
        "labels": {"background": 0, "vessel": 1},
        "numTraining": int(n_train),
        "file_ending": ".png",
        "name": ds_name,
        "reference": "",
        "release": "1.0",
        "tensorImageSize": "2D",
        "description": "Task 3 retinal vessel patches exported from vessel repo",
    }
    with open(ds_dir / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=2)

    splits = [{"train": sorted(train_ids), "val": sorted(val_ids)}]
    with open(ds_dir / "splits_final.json", "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    meta_out = {
        "config": str(cfg_path.resolve()),
        "split_json": str((root / split_json).resolve()) if split_json else None,
        "n_train_patches": len(df_tr),
        "n_val_patches": len(df_va),
        "n_cases": len(case_rows),
        "note": "Evaluate with same protocol as train_task3 after nnUNetv2_predict (map back to patch names if needed).",
    }
    with open(ds_dir / "export_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2)

    print(f"Wrote {ds_dir}")
    print(
        f"  cases={len(seen)}  splits_final train={len(train_ids)} val={len(val_ids)}  "
        f"patches train={len(df_tr)} val={len(df_va)}"
    )


if __name__ == "__main__":
    main()
