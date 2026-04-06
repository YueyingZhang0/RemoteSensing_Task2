#!/usr/bin/env python3
"""tables/efficiency_table.{csv,md}: params, inference ms/patch, peak GPU memory."""
from __future__ import annotations

import csv
import statistics
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import os

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pandas as pd
from src.datasets.seg_sce_patch_dataset import SegSCEPatchDataset
from src.main.train_task3 import _apply_hard_labels
from src.models.task3_unet import UNetBaseline, UNetBaselineJointDifficulty
from src.training.splits import load_or_create_group_split
from src.training.task3_engine import collate_seg_sce
from src.utils.io import load_yaml


def count_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def benchmark_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    warmup: int = 2,
) -> tuple[float, float | None]:
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    for _ in range(warmup):
        with torch.no_grad():
            for batch in loader:
                img = batch["image"].to(device)
                out = model(img)
                logits = out[0] if isinstance(out, tuple) else out
                _ = logits.mean()
                break
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    times = []
    n = 0
    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device)
            bs = img.shape[0]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            out = model(img)
            logits = out[0] if isinstance(out, tuple) else out
            _ = logits.mean()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0 / bs)
            n += bs
    ms_mean = float(statistics.mean(times)) if times else float("nan")
    peak_mb = None
    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    return ms_mean, peak_mb


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base = UNetBaseline(in_channels=3, base=32).to(device)
    joint = UNetBaselineJointDifficulty(in_channels=3, base=32).to(device)

    rows = [
        {
            "method": "P0_baseline",
            "train_wall_clock_sec": "N/A",
            "notes_train_time": "Not logged in artifacts; fill manually if needed.",
            "params_total": count_params(base),
            "inference_ms_per_patch_mean": "N/A",
            "inference_peak_gpu_mb": "N/A",
            "deploy_inference": "UNet only",
        },
        {
            "method": "P5_oracle_matched_trigger",
            "train_wall_clock_sec": "N/A",
            "notes_train_time": "Same architecture as P0; oracle weights use GT scores at train time only.",
            "params_total": count_params(base),
            "inference_ms_per_patch_mean": "N/A",
            "inference_peak_gpu_mb": "N/A",
            "deploy_inference": "UNet only (same as P0)",
        },
        {
            "method": "J3_joint_detach_false",
            "train_wall_clock_sec": "N/A",
            "notes_train_time": "Not logged in artifacts.",
            "params_total": count_params(joint),
            "inference_ms_per_patch_mean": "N/A",
            "inference_peak_gpu_mb": "N/A",
            "deploy_inference": "Forward UNet; use logits output only — difficulty head can be omitted for deployment (tiny overhead if kept).",
        },
    ]

    ms_b, pk_b = benchmark_inference(base, val_loader, device)
    ms_j, pk_j = benchmark_inference(joint, val_loader, device)
    rows[0]["inference_ms_per_patch_mean"] = f"{ms_b:.4f}"
    rows[0]["inference_peak_gpu_mb"] = f"{pk_b:.2f}" if pk_b is not None else "N/A"
    rows[1]["inference_ms_per_patch_mean"] = f"{ms_b:.4f}"
    rows[1]["inference_peak_gpu_mb"] = f"{pk_b:.2f}" if pk_b is not None else "N/A"
    rows[2]["inference_ms_per_patch_mean"] = f"{ms_j:.4f}"
    rows[2]["inference_peak_gpu_mb"] = f"{pk_j:.2f}" if pk_j is not None else "N/A"

    out_dir = ROOT / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "efficiency_table.csv"
    md_path = out_dir / "efficiency_table.md"
    keys = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    md = [
        "## Efficiency (measured on validation loader, same batch size as training)",
        "",
        f"- Device: `{device}`",
        "- Inference: mean wall time per patch (ms) over batches, after warmup.",
        "- P0 and P5 share the same UNet; one timing reported for both.",
        "",
        "| " + " | ".join(keys) + " |",
        "| " + " | ".join(["---"] * len(keys)) + " |",
    ]
    for r in rows:
        md.append("| " + " | ".join(str(r[k]) for k in keys) + " |")
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
