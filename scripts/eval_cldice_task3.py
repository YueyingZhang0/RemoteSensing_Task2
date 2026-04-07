#!/usr/bin/env python3
"""Evaluate mean clDice on val set for P0, P5, J3, Attention U-Net, Swin U-Net. Writes tables/ and cldice_summary.json."""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from src.datasets.seg_sce_patch_dataset import SegSCEPatchDataset
from src.main.train_task3 import _apply_hard_labels
from src.metrics.cldice import cldice_score
from src.models.task3_seg_factory import build_task3_seg_model
from src.models.task3_unet import UNetBaseline, UNetBaselineJointDifficulty
from src.training.splits import load_or_create_group_split
from src.training.task3_engine import collate_seg_sce
from src.utils.io import load_yaml


def load_model(ckpt: Path, kind: str, device: torch.device, *, image_size: int = 128) -> torch.nn.Module:
    if kind == "joint":
        m = UNetBaselineJointDifficulty(in_channels=3, base=32).to(device)
    elif kind == "baseline":
        m = UNetBaseline(in_channels=3, base=32).to(device)
    elif kind == "attention_unet":
        m = build_task3_seg_model("attention_unet", in_channels=3, image_size=image_size).to(device)
    elif kind == "swin_unet":
        m = build_task3_seg_model("swin_unet", in_channels=3, image_size=image_size).to(device)
    else:
        raise ValueError(f"Unknown model kind={kind!r}")
    state = torch.load(ckpt, map_location=device, weights_only=False)
    m.load_state_dict(state["model_state_dict"])
    m.eval()
    return m


@torch.no_grad()
def collect_cldice(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    hard_mask: list[bool] | None = None,
) -> tuple[float, list[float]]:
    scores: list[float] = []
    idx = 0
    for batch in loader:
        img = batch["image"].to(device)
        m = batch["mask"].to(device)
        out = model(img)
        logits = out[0] if isinstance(out, tuple) else out
        prob = torch.sigmoid(logits)
        pred = (prob >= threshold).float().cpu().numpy()
        gt = m.cpu().numpy()
        b = pred.shape[0]
        for i in range(b):
            use = True if hard_mask is None else hard_mask[idx]
            idx += 1
            if not use:
                continue
            p = pred[i, 0].astype(bool)
            g = gt[i, 0] >= 0.5
            scores.append(cldice_score(p, g))
    if not scores:
        return float("nan"), []
    return float(np.mean(scores)), scores


def main() -> None:
    cfg_path = ROOT / "configs" / "task3_J3_joint_detach_false.yaml"
    cfg = load_yaml(cfg_path)
    meta = ROOT / cfg["metadata_csv"]
    img_dir = ROOT / cfg["patch_image_dir"]
    msk_dir = ROOT / cfg["patch_mask_dir"]
    target_col = cfg.get("target_col", "sci_res2_norm")
    group_col = cfg.get("group_col", "sample_id")
    split_json = cfg.get("split_json")
    isz = int(cfg.get("image_size", 128))
    bs = int(cfg.get("batch_size", 16))

    df = pd.read_csv(meta)
    df_tr, df_va = load_or_create_group_split(
        df, group_col, float(cfg.get("val_ratio", 0.2)),
        int(cfg.get("random_state", 42)), split_json, ROOT,
    )
    df_tr, df_va = _apply_hard_labels(df_tr, df_va, target_col, float(cfg.get("hard_ratio", 0.2)))

    val_ds = SegSCEPatchDataset(df_va, img_dir, msk_dir, target_col, isz, augment=False)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0, collate_fn=collate_seg_sce)

    hard_list: list[bool] = []
    for batch in val_loader:
        hard_list.extend(bool(h) for h in batch["is_hard"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    th05 = 0.5

    runs = [
        ("P0_baseline", ROOT / "outputs" / "task3_v2_1_baseline_same_split" / "best_model.pt", "baseline"),
        ("P5_oracle_matched_trigger", ROOT / "outputs" / "task3_v3_1_oracle_matched_trigger_same_split" / "best_model.pt", "baseline"),
        ("J3_joint_detach_false", ROOT / "outputs" / "task3_J3_joint_detach_false" / "best_model.pt", "joint"),
        (
            "attention_unet_baseline",
            ROOT / "outputs" / "task3_attention_unet_baseline" / "best_model.pt",
            "attention_unet",
        ),
        (
            "swin_unet_baseline",
            ROOT / "outputs" / "task3_swin_unet_baseline" / "best_model.pt",
            "swin_unet",
        ),
    ]

    rows = []
    summary: dict = {"threshold_fixed": th05, "methods": {}}
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0, collate_fn=collate_seg_sce)

    for name, ckpt, kind in runs:
        if not ckpt.is_file():
            rows.append({"method": name, "cldice_global_0.5": "N/A", "cldice_hard_0.5": "N/A", "note": "checkpoint missing"})
            summary["methods"][name] = {"error": "checkpoint missing"}
            continue
        model = load_model(ckpt, kind, device, image_size=isz)
        g_mean, _ = collect_cldice(model, val_loader, device, th05, None)
        h_mean, _ = collect_cldice(model, val_loader, device, th05, hard_list)
        rows.append(
            {
                "method": name,
                "cldice_global_0.5": round(g_mean, 6),
                "cldice_hard_0.5": round(h_mean, 6),
                "note": "",
            }
        )
        summary["methods"][name] = {"cldice_global_0.5": g_mean, "cldice_hard_0.5": h_mean}

    summary["narrative"] = (
        "clDice is supplementary; rankings may differ from primary hard-Dice endpoint. "
        "Hard subset = is_hard on val (top sci_res2_norm quantile)."
    )

    tables = ROOT / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    csv_path = tables / "cldice_results.csv"
    md_path = tables / "cldice_results.md"
    keys = ["method", "cldice_global_0.5", "cldice_hard_0.5", "note"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    lines = ["## clDice (binary, skeleton + dilation)", "", f"Fixed threshold = {th05}.", ""]
    lines.append("| " + " | ".join(keys) + " |")
    lines.append("| " + " | ".join(["---"] * len(keys)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(str(r[k]) for k in keys) + " |")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    json_path = ROOT / "cldice_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {csv_path}, {md_path}, {json_path}")


if __name__ == "__main__":
    main()
