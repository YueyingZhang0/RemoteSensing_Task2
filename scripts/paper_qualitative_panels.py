#!/usr/bin/env python3
"""Qualitative figures: J3 vs P0 vs P5; manifest + combined panel."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.morphology import skeletonize
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.datasets.seg_sce_patch_dataset import SegSCEPatchDataset
from src.main.train_task3 import _apply_hard_labels
from src.models.task3_unet import UNetBaseline, UNetBaselineJointDifficulty
from src.training.splits import load_or_create_group_split
from src.training.task3_engine import collate_seg_sce
from src.utils.io import load_yaml


def load_model(ckpt: Path, kind: str, device: torch.device) -> torch.nn.Module:
    if kind == "joint":
        m = UNetBaselineJointDifficulty(in_channels=3, base=32).to(device)
    else:
        m = UNetBaseline(in_channels=3, base=32).to(device)
    s = torch.load(ckpt, map_location=device, weights_only=False)
    m.load_state_dict(s["model_state_dict"])
    m.eval()
    return m


@torch.no_grad()
def pred_mask(
    model: torch.nn.Module,
    img_t: torch.Tensor,
    device: torch.device,
    thresh: float,
) -> np.ndarray:
    x = img_t.unsqueeze(0).to(device)
    out = model(x)
    logits = out[0] if isinstance(out, tuple) else out
    p = torch.sigmoid(logits)[0, 0].cpu().numpy()
    return (p >= thresh).astype(np.float32)


def error_rgb(gt: np.ndarray, pr: np.ndarray) -> np.ndarray:
    """H,W,3: FN red, FP blue, TP white-ish."""
    g = gt > 0.5
    p = pr > 0.5
    h, w = g.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[..., 0] = (g & ~p).astype(np.float32)
    rgb[..., 2] = (~g & p).astype(np.float32)
    rgb[..., 1] = (g & p).astype(np.float32) * 0.35
    return np.clip(rgb, 0, 1)


def skel_overlay(rgb: np.ndarray, pred_bin: np.ndarray, gt_bin: np.ndarray) -> np.ndarray:
    out = rgb.copy().astype(np.float32)
    sp = skeletonize(pred_bin > 0.5)
    sg = skeletonize(gt_bin > 0.5)
    out[sg] = out[sg] * 0.5 + np.array([0.0, 1.0, 0.0], dtype=np.float32) * 0.5
    out[sp] = out[sp] * 0.5 + np.array([1.0, 0.0, 1.0], dtype=np.float32) * 0.5
    return np.clip(out, 0, 1)


def pick_cases(audit: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
    hard = audit[audit["is_hard"]]
    c1 = hard[(hard["delta_dice_j3_vs_p0"] > 0.02) & (hard["delta_dice_j3_vs_p5"] > 0.005)]
    if len(c1) == 0:
        c1 = hard.nlargest(1, "delta_dice_j3_vs_p0")
    case1 = c1.loc[c1["delta_dice_j3_vs_p0"].idxmax()]

    c2 = hard[np.abs(hard["delta_dice_j3_vs_p5"]) < 0.004]
    if len(c2) == 0:
        c2 = hard.assign(ad=np.abs(hard["delta_dice_j3_vs_p5"])).nsmallest(3, "ad")
    case2 = c2.iloc[0]

    c3 = audit[audit["delta_dice_j3_vs_p0"] < -0.005]
    if len(c3) == 0:
        c3 = audit.nsmallest(1, "delta_dice_j3_vs_p0")
    c3h = c3[c3["is_hard"]]
    if len(c3h) > 0:
        case3 = c3h.loc[c3h["delta_dice_j3_vs_p0"].idxmin()]
    else:
        case3 = c3.loc[c3["delta_dice_j3_vs_p0"].idxmin()]
    return case1, case2, case3


def render_case_row(
    case: pd.Series,
    img_path: Path,
    msk_path: Path,
    models: Dict[str, torch.nn.Module],
    device: torch.device,
    img_t: torch.Tensor,
    thresh: float,
    title: str,
) -> None:
    rgb = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
    gt = np.asarray(Image.open(msk_path).convert("L"), dtype=np.float32) / 255.0
    gt_bin = (gt > 0.5).astype(np.float32)

    preds = {}
    for k, m in models.items():
        preds[k] = pred_mask(m, img_t, device, thresh)

    cols = ["Image", "GT", "P0", "P5", "J3", "Error(J3)", "Skel(J3)"]
    fig, axes = plt.subplots(1, len(cols), figsize=(14, 2.4))
    axes[0].imshow(np.clip(rgb, 0, 1))
    axes[0].set_title("RGB")
    axes[1].imshow(gt_bin, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("GT")
    axes[2].imshow(preds["p0"], cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("P0")
    axes[3].imshow(preds["p5"], cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("P5")
    axes[4].imshow(preds["j3"], cmap="gray", vmin=0, vmax=1)
    axes[4].set_title("J3")
    axes[5].imshow(error_rgb(gt_bin, preds["j3"]))
    axes[5].set_title("Err J3")
    axes[6].imshow(skel_overlay(rgb, preds["j3"], gt_bin))
    axes[6].set_title("Skel")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=10, y=1.05)
    plt.tight_layout()


def main() -> None:
    cfg_path = ROOT / "configs" / "task3_J3_joint_detach_false.yaml"
    cfg = load_yaml(cfg_path)
    meta = Path(cfg["metadata_csv"])
    if not meta.is_absolute():
        meta = ROOT / meta
    img_dir = Path(cfg["patch_image_dir"])
    if not img_dir.is_absolute():
        img_dir = ROOT / img_dir
    msk_dir = Path(cfg["patch_mask_dir"])
    if not msk_dir.is_absolute():
        msk_dir = ROOT / msk_dir
    target_col = cfg.get("target_col", "sci_res2_norm")
    group_col = cfg.get("group_col", "sample_id")
    split_json = cfg.get("split_json")
    isz = int(cfg.get("image_size", 128))

    audit_path = ROOT / "analysis" / "j3_patch_audit.csv"
    if not audit_path.is_file():
        raise SystemExit("Run: python -m src.analysis.j3_patch_audit")

    audit = pd.read_csv(audit_path)
    case1, case2, case3 = pick_cases(audit)

    df = pd.read_csv(meta)
    df_tr, df_va = load_or_create_group_split(
        df, group_col, float(cfg.get("val_ratio", 0.2)),
        int(cfg.get("random_state", 42)), split_json, ROOT,
    )
    df_tr, df_va = _apply_hard_labels(df_tr, df_va, target_col, float(cfg.get("hard_ratio", 0.2)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {
        "p0": load_model(ROOT / "outputs" / "task3_v2_1_baseline_same_split" / "best_model.pt", "baseline", device),
        "p5": load_model(ROOT / "outputs" / "task3_v3_1_oracle_matched_trigger_same_split" / "best_model.pt", "baseline", device),
        "j3": load_model(ROOT / "outputs" / "task3_J3_joint_detach_false" / "best_model.pt", "joint", device),
    }

    val_ds = SegSCEPatchDataset(df_va, img_dir, msk_dir, target_col, isz, augment=False)
    name_to_tensor: Dict[str, torch.Tensor] = {}
    for i in range(len(val_ds)):
        s = val_ds[i]
        name_to_tensor[str(s["patch_name"])] = s["image"]

    fig_dir = ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    thresh = 0.5
    manifest: Dict[str, Any] = {"threshold": thresh, "cases": []}

    def one_case(case: pd.Series, idx: int, tag: str) -> None:
        pn = str(case["patch_name"])
        ip = Path(img_dir) / pn
        mp = Path(msk_dir) / pn
        img_t = name_to_tensor[pn]
        title = (
            f"Case {idx} ({tag}): {pn} | hard={bool(case['is_hard'])} "
            f"| d(J3-P0)={case['delta_dice_j3_vs_p0']:.4f} d(J3-P5)={case['delta_dice_j3_vs_p5']:.4f}"
        )
        render_case_row(case, ip, mp, models, device, img_t, thresh, title)
        out = fig_dir / f"qualitative_case_{idx:02d}.png"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        manifest["cases"].append(
            {
                "file": str(out.relative_to(ROOT)),
                "patch_name": pn,
                "sample_id": str(case.get("sample_id", "")),
                "is_true_hard": bool(case["is_hard"]),
                "gt_sci_res2_norm": float(case.get("gt_sci_res2_norm", case.get("sci_res2_norm", float("nan")))),
                "p0_dice": float(case["p0_dice"]),
                "p5_dice": float(case["p5_dice"]),
                "j3_dice": float(case["j3_dice"]),
                "delta_j3_vs_p0": float(case["delta_dice_j3_vs_p0"]),
                "delta_j3_vs_p5": float(case["delta_dice_j3_vs_p5"]),
                "tag": tag,
            }
        )

    one_case(case1, 1, "J3_clear_win_hard")
    one_case(case2, 2, "J3_near_P5_hard")
    one_case(case3, 3, "J3_failure")

    fig_all, axes_all = plt.subplots(3, 1, figsize=(14, 7.2))
    for row_i, (case, tag) in enumerate([(case1, "1"), (case2, "2"), (case3, "3")]):
        pn = str(case["patch_name"])
        ip = Path(img_dir) / pn
        mp = Path(msk_dir) / pn
        rgb = np.asarray(Image.open(ip).convert("RGB"), dtype=np.float32) / 255.0
        gt = np.asarray(Image.open(mp).convert("L"), dtype=np.float32) / 255.0
        gt_bin = (gt > 0.5).astype(np.float32)
        img_t = name_to_tensor[pn]
        pr_j3 = pred_mask(models["j3"], img_t, device, thresh)
        ax = axes_all[row_i]
        comp = np.concatenate([np.clip(rgb, 0, 1), np.repeat(gt_bin[..., None], 3, axis=2)], axis=1)
        p0s = pred_mask(models["p0"], img_t, device, thresh)
        p5s = pred_mask(models["p5"], img_t, device, thresh)
        mid = np.concatenate(
            [
                np.repeat(p0s[..., None], 3, axis=2),
                np.repeat(p5s[..., None], 3, axis=2),
                np.repeat(pr_j3[..., None], 3, axis=2),
            ],
            axis=1,
        )
        ax.imshow(np.concatenate([comp, mid], axis=1))
        ax.set_ylabel(f"Row {tag}")
        ax.axis("off")
    axes_all[0].set_title("RGB|GT  ||  P0 | P5 | J3  (per row: case 1–3)")
    plt.tight_layout()
    p_main = fig_dir / "qualitative_panel_main.png"
    fig_all.savefig(p_main, dpi=180, bbox_inches="tight")
    plt.close()

    man_path = fig_dir / "qualitative_case_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {p_main}, qualitative_case_01–03.png, {man_path}")


if __name__ == "__main__":
    main()
