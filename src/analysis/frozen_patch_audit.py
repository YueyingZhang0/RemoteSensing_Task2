#!/usr/bin/env python3
"""Per-patch audit: baseline vs frozen seg, probe preds, oracle trigger, topology merge, 4-way groups, example exports."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.datasets.seg_sce_patch_dataset import SegSCEPatchDataset
from src.main.train_task3 import _apply_hard_labels
from src.models.task3_unet import UNetBaseline
from src.training.splits import load_or_create_group_split
from src.training.task3_engine import _per_sample_dice_recall, collate_seg_sce, load_frozen_sce_probe
from src.utils.io import load_yaml, save_json


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/task3_v3_1_predicted_frozen.yaml")
    p.add_argument("--baseline_ckpt", type=str, default="outputs/task3_v2_1_baseline_same_split/best_model.pt")
    p.add_argument("--frozen_ckpt", type=str, default="outputs/task3_v3_1_predicted_frozen_same_split/best_model.pt")
    p.add_argument("--probe_ckpt", type=str, default="outputs/task3_difficulty_probe/best_sce_branch.pt")
    p.add_argument("--tau", type=float, default=0.6, help="Threshold for pred_above_tau and oracle_above_tau")
    p.add_argument("--examples_per_group", type=int, default=10)
    p.add_argument(
        "--out_dir",
        type=str,
        default="outputs/task3_v3_1_predicted_frozen_same_split",
    )
    args = p.parse_args()

    root = Path(__file__).resolve().parents[2]
    cfg = load_yaml(root / args.config)
    target_col = cfg.get("target_col", "sci_res2_norm")
    group_col = cfg.get("group_col", "sample_id")
    hard_ratio = float(cfg.get("hard_ratio", 0.2))
    val_ratio = float(cfg.get("val_ratio", 0.2))
    random_state = int(cfg.get("random_state", 42))
    split_json = cfg.get("split_json")
    bs = int(cfg.get("batch_size", 16))
    nw = int(cfg.get("num_workers", 0))
    isz = int(cfg.get("image_size", 128))

    meta = Path(cfg["metadata_csv"])
    if not meta.is_absolute():
        meta = root / meta
    img_dir = Path(cfg["patch_image_dir"])
    msk_dir = Path(cfg["patch_mask_dir"])
    if not img_dir.is_absolute():
        img_dir = root / img_dir
    if not msk_dir.is_absolute():
        msk_dir = root / msk_dir

    df = pd.read_csv(meta)
    sj = Path(split_json) if split_json else None
    if sj and not sj.is_absolute():
        sj = root / sj
    df_tr, df_va = load_or_create_group_split(df, group_col, val_ratio, random_state, str(sj) if sj else None, root)
    df_tr, df_va = _apply_hard_labels(df_tr, df_va, target_col, hard_ratio)

    val_ds = SegSCEPatchDataset(df_va, img_dir, msk_dir, target_col, isz, augment=False)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, collate_fn=collate_seg_sce)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base = UNetBaseline(in_channels=3, base=32).to(device)
    frz = UNetBaseline(in_channels=3, base=32).to(device)
    bpath = Path(args.baseline_ckpt)
    fpath = Path(args.frozen_ckpt)
    if not bpath.is_absolute():
        bpath = root / bpath
    if not fpath.is_absolute():
        fpath = root / fpath
    base.load_state_dict(torch.load(bpath, map_location=device)["model_state_dict"])
    frz.load_state_dict(torch.load(fpath, map_location=device)["model_state_dict"])
    base.eval()
    frz.eval()

    ppath = Path(args.probe_ckpt)
    if not ppath.is_absolute():
        ppath = root / ppath
    probe = load_frozen_sce_probe(ppath, device, in_channels=3, base_channels=32, mlp_hidden=64, dropout=0.0)
    probe.eval()

    meta_by_name = df.set_index("patch_name", drop=False)

    rows: List[Dict[str, Any]] = []
    tau = float(args.tau)

    with torch.no_grad():
        for batch in val_loader:
            img = batch["image"].to(device)
            m = batch["mask"].to(device)
            gt_sci = batch["sci_res2_norm"].cpu().numpy().ravel()
            names = batch["patch_name"]
            is_hard = batch["is_hard"]

            lb = base(img)
            lf = frz(img)
            d_b, r_b = _per_sample_dice_recall(lb, m, 0.5)
            d_f, r_f = _per_sample_dice_recall(lf, m, 0.5)

            pr = probe(img)["sci_pred"].squeeze(1).clamp(0.0, 1.0).cpu().numpy().ravel()

            for i, name in enumerate(names):
                row_m: Dict[str, Any] = {
                    "patch_name": name,
                    "gt_sci_res2_norm": float(gt_sci[i]),
                    "pred_sci": float(pr[i]),
                    "is_true_hard": bool(is_hard[i]),
                    "pred_above_tau": bool(pr[i] > tau),
                    "oracle_above_tau": bool(gt_sci[i] > tau),
                    "baseline_dice": float(d_b[i]),
                    "frozen_dice": float(d_f[i]),
                    "delta_dice_frozen_vs_baseline": float(d_f[i] - d_b[i]),
                    "baseline_recall": float(r_b[i]),
                    "frozen_recall": float(r_f[i]),
                    "delta_recall_frozen_vs_baseline": float(r_f[i] - r_b[i]),
                }
                if name in meta_by_name.index:
                    mr = meta_by_name.loc[name]
                    if isinstance(mr, pd.DataFrame):
                        mr = mr.iloc[0]
                    row_m["sample_id"] = mr.get(group_col, "")
                    for col in (
                        "fov_ratio",
                        "vessel_area",
                        "skeleton_len",
                        "junction_count",
                        "endpoint_count",
                        "component_count",
                        "endpoint_eff",
                        "topo_density",
                        "sci_topo_norm",
                    ):
                        if col in mr.index:
                            v = mr[col]
                            row_m[col] = float(v) if pd.notna(v) else None
                        else:
                            row_m[col] = None
                else:
                    row_m["sample_id"] = ""
                rows.append(row_m)

    audit_df = pd.DataFrame(rows)

    def _group_tag(r: pd.Series) -> str:
        p, o = bool(r["pred_above_tau"]), bool(r["oracle_above_tau"])
        if p and not o:
            return "pred_only"
        if o and not p:
            return "oracle_only"
        if p and o:
            return "both"
        return "neither"

    audit_df["group"] = audit_df.apply(_group_tag, axis=1)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "frozen_patch_audit.csv"
    audit_df.to_csv(csv_path, index=False)

    # Group summaries
    agg_cols = {
        "gt_sci_res2_norm": "mean",
        "pred_sci": "mean",
        "delta_dice_frozen_vs_baseline": "mean",
        "delta_recall_frozen_vs_baseline": "mean",
    }
    if "fov_ratio" in audit_df.columns:
        agg_cols["fov_ratio"] = "mean"
    if "vessel_area" in audit_df.columns:
        agg_cols["vessel_area"] = "mean"

    summaries: List[Dict[str, Any]] = []
    for g in ["pred_only", "oracle_only", "both", "neither"]:
        sub = audit_df[audit_df["group"] == g]
        entry: Dict[str, Any] = {"group": g, "n_patches": int(len(sub))}
        if len(sub):
            for k, how in agg_cols.items():
                if k in sub.columns:
                    entry[f"mean_{k}"] = float(sub[k].mean())
        summaries.append(entry)

    summ_path = out_dir / "frozen_patch_groups_summary.json"
    save_json({"tau": tau, "groups": summaries}, summ_path)
    pd.DataFrame(summaries).to_csv(out_dir / "frozen_patch_groups_summary.csv", index=False)

    # Representative patch images
    ex_dir = out_dir / "patch_group_examples"
    ex_dir.mkdir(parents=True, exist_ok=True)
    n_ex = int(args.examples_per_group)
    for g in ["pred_only", "oracle_only", "both", "neither"]:
        sub = audit_df[audit_df["group"] == g]
        gdir = ex_dir / g
        gdir.mkdir(parents=True, exist_ok=True)
        if len(sub) == 0:
            continue
        sub2 = sub.copy()
        sub2["abs_delta_dice"] = sub2["delta_dice_frozen_vs_baseline"].abs()
        top = sub2.nlargest(min(n_ex, len(sub2)), "abs_delta_dice")
        for _, r in top.iterrows():
            src = img_dir / str(r["patch_name"])
            if src.is_file():
                dst = gdir / str(r["patch_name"])
                shutil.copy2(src, dst)

    print(f"Wrote {csv_path}")
    print(f"Wrote {summ_path}")
    print(f"Examples under {ex_dir}")


if __name__ == "__main__":
    main()
