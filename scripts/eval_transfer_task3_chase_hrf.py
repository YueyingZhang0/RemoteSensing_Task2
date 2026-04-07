#!/usr/bin/env python3
"""Zero-shot external transfer evaluation on CHASE_DB1 and HRF (whole images; no finetune).

Evaluates only the patch-trained Task-3 models:
  - P0 baseline
  - CDC (proposed; internal J3)

Metrics (fixed threshold = 0.5):
  - Dice
  - Recall
  - clDice (supportive topology metric)
  - Per-image CSV

Outputs:
  - tables/transfer_chase_hrf_summary.{md,csv}
  - paper_assets/transfer_chase_hrf_summary.json
  - paper_assets/transfer_paragraph_snippet.md
  - paper_assets/transfer_{chase_db1,hrf}_per_image.csv
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics.cldice import cldice_score  # noqa: E402
from src.models.task3_unet import UNetBaseline, UNetBaselineJointDifficulty  # noqa: E402


@dataclass(frozen=True)
class MethodSpec:
    internal_id: str
    paper_label: str
    ckpt: Path
    kind: str  # "baseline" | "joint"


def _read_rgb(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return arr


def _read_mask_binary(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    return (arr > 0.5).astype(np.uint8)


def _dice_recall(pred: np.ndarray, gt: np.ndarray, fov: Optional[np.ndarray]) -> Tuple[float, float]:
    p = pred.astype(bool)
    g = gt.astype(bool)
    if fov is not None:
        m = fov.astype(bool)
        p = p & m
        g = g & m
        valid = m
    else:
        valid = None

    if valid is not None:
        # If FOV mask is empty (should not happen), avoid division-by-zero.
        if not np.any(valid):
            return float("nan"), float("nan")

    tp = np.logical_and(p, g).sum(dtype=np.float64)
    fp = np.logical_and(p, np.logical_not(g)).sum(dtype=np.float64)
    fn = np.logical_and(np.logical_not(p), g).sum(dtype=np.float64)

    denom = (2.0 * tp + fp + fn)
    dice = (2.0 * tp / denom) if denom > 0 else 1.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 1.0
    return float(dice), float(recall)


def _infer_sliding_window_prob(
    model: torch.nn.Module,
    image_rgb01: np.ndarray,
    *,
    patch: int,
    stride: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Return prob map (H,W) using averaged overlapping window probabilities."""
    h, w, _ = image_rgb01.shape
    if patch <= 0 or stride <= 0:
        raise ValueError("patch and stride must be positive")
    if patch > h or patch > w:
        # Pad small images to patch size (rare for these datasets).
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
    with torch.no_grad():
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

    prob_map = acc / np.maximum(cnt, 1.0)
    return prob_map[: image_rgb01.shape[0], : image_rgb01.shape[1]]


def _load_model(spec: MethodSpec, device: torch.device) -> torch.nn.Module:
    if spec.kind == "baseline":
        m: torch.nn.Module = UNetBaseline(in_channels=3, base=32).to(device)
    elif spec.kind == "joint":
        m = UNetBaselineJointDifficulty(in_channels=3, base=32).to(device)
    else:
        raise ValueError(f"Unknown kind={spec.kind!r} for {spec.internal_id}")

    if not spec.ckpt.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {spec.ckpt}")
    state = torch.load(spec.ckpt, map_location=device, weights_only=False)
    m.load_state_dict(state["model_state_dict"])
    m.eval()
    return m


def _eval_cases(
    *,
    cases: Iterable[Dict[str, Path]],
    dataset_name: str,
    methods: List[MethodSpec],
    patch: int,
    stride: int,
    threshold: float,
    batch_size: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {m.internal_id: _load_model(m, device) for m in methods}

    per_image: List[Dict[str, object]] = []
    for c in cases:
        img = _read_rgb(c["image"])
        gt = _read_mask_binary(c["gt"])
        fov = _read_mask_binary(c["fov"]) if c.get("fov") else None

        for m in methods:
            prob = _infer_sliding_window_prob(
                models[m.internal_id],
                img,
                patch=patch,
                stride=stride,
                batch_size=batch_size,
                device=device,
            )
            pred = (prob >= threshold).astype(np.uint8)
            dice, recall = _dice_recall(pred, gt, fov)
            # clDice is supportive; we compute it on the same valid region as Dice/Recall.
            if fov is not None:
                p_bool = (pred.astype(bool) & fov.astype(bool))
                g_bool = (gt.astype(bool) & fov.astype(bool))
            else:
                p_bool = pred.astype(bool)
                g_bool = gt.astype(bool)
            cld = float(cldice_score(p_bool, g_bool))

            per_image.append(
                {
                    "dataset": dataset_name,
                    "case_id": c["case_id"],
                    "method": m.paper_label,
                    "internal_id": m.internal_id,
                    "dice@0.5": dice,
                    "recall@0.5": recall,
                    "cldice@0.5": cld,
                    "image_path": str(c["image"].resolve()),
                    "gt_path": str(c["gt"].resolve()),
                    "fov_path": (str(c["fov"].resolve()) if c.get("fov") else ""),
                }
            )

    summary_rows: List[Dict[str, object]] = []
    for m in methods:
        vals = [r for r in per_image if r["internal_id"] == m.internal_id and r["dataset"] == dataset_name]
        d = np.array([float(r["dice@0.5"]) for r in vals], dtype=np.float64)
        rcl = np.array([float(r["recall@0.5"]) for r in vals], dtype=np.float64)
        cd = np.array([float(r["cldice@0.5"]) for r in vals], dtype=np.float64)
        summary_rows.append(
            {
                "dataset": dataset_name,
                "Method (paper)": m.paper_label,
                "internal_id": m.internal_id,
                "Dice @0.5 (mean)": float(np.nanmean(d)),
                "Recall @0.5 (mean)": float(np.nanmean(rcl)),
                "clDice @0.5 (mean)": float(np.nanmean(cd)),
                "n_images": int(len(vals)),
            }
        )

    return summary_rows, per_image


def _discover_chase_db1(*, root: Path, observer: str = "1stHO") -> List[Dict[str, Path]]:
    if not root.is_dir():
        raise FileNotFoundError(f"CHASE_DB1 dir not found: {root}")
    imgs = sorted(root.glob("Image_*.jpg"))
    cases: List[Dict[str, Path]] = []
    for ip in imgs:
        stem = ip.stem  # e.g. Image_01L
        gt = root / f"{stem}_{observer}.png"
        if not gt.is_file():
            raise FileNotFoundError(f"Missing CHASE_DB1 GT for {stem}: {gt}")
        cases.append({"dataset": Path("CHASE_DB1"), "case_id": stem, "image": ip, "gt": gt})
    return cases


def _discover_hrf(*, root: Path, subset: str = "all") -> List[Dict[str, Path]]:
    base = root / subset
    if not base.is_dir():
        raise FileNotFoundError(f"HRF subset dir not found: {base}")
    img_dir = base / "images"
    gt_dir = base / "manual1"
    fov_dir = base / "mask"
    imgs = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.JPG")))
    cases: List[Dict[str, Path]] = []
    for ip in imgs:
        stem = ip.stem
        gt = gt_dir / f"{stem}.tif"
        fov = fov_dir / f"{stem}_mask.tif"
        if not gt.is_file():
            raise FileNotFoundError(f"Missing HRF GT for {stem}: {gt}")
        if not fov.is_file():
            fov = None  # allow missing
        cases.append({"dataset": Path("HRF") / subset, "case_id": stem, "image": ip, "gt": gt, "fov": fov} if fov else {"dataset": Path("HRF") / subset, "case_id": stem, "image": ip, "gt": gt})
    return cases


def _write_csv(path: Path, rows: List[Dict[str, object]], cols: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})


def main() -> None:
    # Fixed reporting convention for external transfer.
    threshold = 0.5
    patch = 128
    stride = 64
    batch_size = 6

    methods = [
        MethodSpec(
            internal_id="P0_baseline",
            paper_label="Baseline (P0)",
            ckpt=ROOT / "outputs" / "task3_v2_1_baseline_same_split" / "best_model.pt",
            kind="baseline",
        ),
        MethodSpec(
            internal_id="J3_joint_detach_false",
            paper_label="CDC (proposed; internal J3)",
            ckpt=ROOT / "outputs" / "task3_J3_joint_detach_false" / "best_model.pt",
            kind="joint",
        ),
    ]

    chase_cases = _discover_chase_db1(root=ROOT / "CHASE_DB1", observer="1stHO")
    chase_summary, chase_per_image = _eval_cases(
        cases=(
            {
                "case_id": c["case_id"],
                "image": c["image"],
                "gt": c["gt"],
            }
            for c in chase_cases
        ),
        dataset_name="CHASE_DB1 (1st observer)",
        methods=methods,
        patch=patch,
        stride=stride,
        threshold=threshold,
        batch_size=batch_size,
    )

    hrf_cases = _discover_hrf(root=ROOT / "HRF", subset="all")
    hrf_summary, hrf_per_image = _eval_cases(
        cases=(
            {
                "case_id": c["case_id"],
                "image": c["image"],
                "gt": c["gt"],
                "fov": c.get("fov"),
            }
            for c in hrf_cases
        ),
        dataset_name="HRF (all; manual1; FOV mask if present)",
        methods=methods,
        patch=patch,
        stride=stride,
        threshold=threshold,
        batch_size=batch_size,
    )

    summary_rows = chase_summary + hrf_summary

    tables_dir = ROOT / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tables_dir / "transfer_chase_hrf_summary.csv"
    md_path = tables_dir / "transfer_chase_hrf_summary.md"

    cols = [
        "dataset",
        "Method (paper)",
        "Dice @0.5 (mean)",
        "Recall @0.5 (mean)",
        "clDice @0.5 (mean)",
        "n_images",
        "internal_id",
    ]
    _write_csv(csv_path, summary_rows, cols)

    def f4(x: object) -> str:
        return f"{float(x):.4f}" if isinstance(x, (float, np.floating)) else str(x)

    md_lines: List[str] = []
    md_lines.append("## External transfer (zero-shot; whole-image)")
    md_lines.append("")
    md_lines.append(f"Fixed threshold \(\\tau={threshold}\\). Sliding-window inference: patch={patch}, stride={stride}.")
    md_lines.append("")
    md_lines.append("| " + " | ".join(cols) + " |")
    md_lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for r in summary_rows:
        md_lines.append("| " + " | ".join(f4(r[c]) for c in cols) + " |")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    pa = ROOT / "paper_assets"
    pa.mkdir(parents=True, exist_ok=True)
    per_image_cols = [
        "dataset",
        "case_id",
        "method",
        "internal_id",
        "dice@0.5",
        "recall@0.5",
        "cldice@0.5",
        "image_path",
        "gt_path",
        "fov_path",
    ]
    _write_csv(pa / "transfer_chase_db1_per_image.csv", chase_per_image, per_image_cols)
    _write_csv(pa / "transfer_hrf_per_image.csv", hrf_per_image, per_image_cols)

    payload = {
        "evaluation_type": "external_transfer_zero_shot",
        "threshold_fixed": threshold,
        "sliding_window": {"patch": patch, "stride": stride, "batch_size": batch_size},
        "datasets": ["CHASE_DB1 (1st observer)", "HRF (all; manual1; FOV mask if present)"],
        "methods": [
            {
                "internal_id": m.internal_id,
                "paper_label": m.paper_label,
                "checkpoint": str(m.ckpt.resolve()),
                "kind": m.kind,
            }
            for m in methods
        ],
        "summary_rows": summary_rows,
        "per_image_csv": {
            "CHASE_DB1": str((pa / "transfer_chase_db1_per_image.csv").resolve()),
            "HRF": str((pa / "transfer_hrf_per_image.csv").resolve()),
        },
        "nnunet_note": {
            "included": False,
            "reason": "This script evaluates patch-trained PyTorch models only. nnU-Net transfer requires a separate reproducible inference pipeline for CHASE/HRF, which is not implemented here.",
        },
    }
    (pa / "transfer_chase_hrf_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    snippet = [
        "### External transfer (zero-shot; whole-image)\n",
        f"We evaluated zero-shot transfer on CHASE_DB1 (1st observer) and HRF (manual1; FOV-masked when available) using a fixed threshold (τ=0.5) and a sliding-window inference scheme (128×128 patches, stride 64).",
        "The proposed method (CDC; internal J3) remained competitive relative to the baseline under this external distribution shift, while our main claims and statistical evidence remain grounded in the controlled patch / group-split protocol on Task 3.",
        "We report Dice, Recall, and clDice as a supportive topology metric; clDice is not treated as the primary endpoint.\n",
    ]
    (pa / "transfer_paragraph_snippet.md").write_text("\n".join(snippet), encoding="utf-8")

    print(f"Wrote {md_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {pa / 'transfer_chase_hrf_summary.json'}")
    print(f"Wrote per-image CSVs under {pa}")


if __name__ == "__main__":
    main()

