#!/usr/bin/env python3
"""Qualitative panels for CHASE_DB1 (thin-vessel proxy overlay + predictions).

Panels per case:
  - RGB
  - GT
  - thin-vessel hard mask overlay
  - P0 zero-shot
  - J3 zero-shot (CDC)
  - P0 fine-tuned
  - J3 fine-tuned

Outputs:
  - figures/ext_adapt_chase_case_*.png
  - figures/ext_adapt_chase_manifest.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics.thinvessel_proxy import ThinVesselProxyConfig, thin_vessel_hard_mask  # noqa: E402
from src.models.task3_unet import UNetBaseline, UNetBaselineJointDifficulty  # noqa: E402


def _read_rgb01(p: Path) -> np.ndarray:
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0


def _read_mask01(p: Path) -> np.ndarray:
    a = np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0
    return (a > 0.5).astype(np.uint8)


def _load_model(kind: str, ckpt: Path, device: torch.device) -> torch.nn.Module:
    if kind == "baseline":
        m: torch.nn.Module = UNetBaseline(in_channels=3, base=32).to(device)
    else:
        m = UNetBaselineJointDifficulty(in_channels=3, base=32).to(device)
    s = torch.load(ckpt, map_location=device, weights_only=False)
    m.load_state_dict(s["model_state_dict"])
    m.eval()
    return m


@torch.no_grad()
def _pred_mask(model: torch.nn.Module, rgb01: np.ndarray, device: torch.device, thresh: float = 0.5) -> np.ndarray:
    """Whole-image prediction via sliding window to avoid size constraints of the U-Net."""
    patch = 128
    stride = 64
    batch_size = 6
    h, w, _ = rgb01.shape
    if patch > h or patch > w:
        pad_h = max(0, patch - h)
        pad_w = max(0, patch - w)
        rgb01 = np.pad(rgb01, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        h, w, _ = rgb01.shape

    ys = list(range(0, max(1, h - patch + 1), stride))
    xs = list(range(0, max(1, w - patch + 1), stride))
    if ys[-1] != h - patch:
        ys.append(h - patch)
    if xs[-1] != w - patch:
        xs.append(w - patch)

    acc = np.zeros((h, w), dtype=np.float32)
    cnt = np.zeros((h, w), dtype=np.float32)
    coords = [(y, x) for y in ys for x in xs]

    model.eval()
    for i in range(0, len(coords), batch_size):
        bc = coords[i : i + batch_size]
        patches = []
        for (y, x) in bc:
            p = rgb01[y : y + patch, x : x + patch, :]
            patches.append(torch.from_numpy(p).permute(2, 0, 1))
        xb = torch.stack(patches, dim=0).to(device=device, dtype=torch.float32)
        out = model(xb)
        logits = out[0] if isinstance(out, tuple) else out
        prob = torch.sigmoid(logits).detach().cpu().numpy()  # (B,1,patch,patch)
        for j, (y, x) in enumerate(bc):
            acc[y : y + patch, x : x + patch] += prob[j, 0]
            cnt[y : y + patch, x : x + patch] += 1.0

    prob_map = acc / np.maximum(cnt, 1.0)
    prob_map = prob_map[: rgb01.shape[0], : rgb01.shape[1]]
    return (prob_map[:h, :w] >= float(thresh)).astype(np.uint8)


def _overlay_hard(rgb01: np.ndarray, hard: np.ndarray) -> np.ndarray:
    out = rgb01.copy()
    m = hard.astype(bool)
    out[m] = out[m] * 0.4 + np.array([1.0, 0.2, 0.2], dtype=np.float32) * 0.6
    return np.clip(out, 0, 1)


def _choose_cases_from_per_image(csv_path: Path) -> List[str]:
    # Prefer: one best improvement, one medium, one failure/mixed using CDC vs P0 dice deltas.
    import pandas as pd

    df = pd.read_csv(csv_path)
    best = df.sort_values("delta", ascending=False).iloc[0]["case_id"]
    mid = df.sort_values("delta", ascending=False).iloc[len(df) // 2]["case_id"]
    worst = df.sort_values("delta", ascending=True).iloc[0]["case_id"]
    out = []
    for c in [best, mid, worst]:
        if c not in out:
            out.append(str(c))
    return out


def _build_delta_csv(p0_csv: Path, j3_csv: Path) -> Path:
    import pandas as pd

    p0 = pd.read_csv(p0_csv)[["case_id", "dice@0.5"]].rename(columns={"dice@0.5": "p0"})
    j3 = pd.read_csv(j3_csv)[["case_id", "dice@0.5"]].rename(columns={"dice@0.5": "j3"})
    m = p0.merge(j3, on="case_id", how="inner")
    m["delta"] = m["j3"] - m["p0"]
    tmp = ROOT / "paper_assets" / "_tmp_chase_ft_delta.csv"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(tmp, index=False)
    return tmp


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    thresh = 0.5

    chase_dir = ROOT / "CHASE_DB1"
    # Use fine-tuned external per-image CSV to select representative test cases.
    p0_csv = ROOT / "outputs" / "ft_chase_p0_seed42" / "external_test_per_image_metrics.csv"
    j3_csv = ROOT / "outputs" / "ft_chase_j3_seed42" / "external_test_per_image_metrics.csv"
    delta_csv = _build_delta_csv(p0_csv, j3_csv)
    cases = _choose_cases_from_per_image(delta_csv)

    models = {
        "P0_zero": _load_model("baseline", ROOT / "outputs" / "task3_v2_1_baseline_same_split" / "best_model.pt", device),
        "J3_zero": _load_model("joint", ROOT / "outputs" / "task3_J3_joint_detach_false" / "best_model.pt", device),
        "P0_ft": _load_model("baseline", ROOT / "outputs" / "ft_chase_p0_seed42" / "best_model.pt", device),
        "J3_ft": _load_model("joint", ROOT / "outputs" / "ft_chase_j3_seed42" / "best_model.pt", device),
    }

    cfg = ThinVesselProxyConfig(r_th=2, dilate_iters=1)
    fig_dir = ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {"dataset": "CHASE_DB1 (1st observer)", "threshold": thresh, "cases": []}

    for idx, cid in enumerate(cases, start=1):
        ip = chase_dir / f"{cid}.jpg"
        gp = chase_dir / f"{cid}_1stHO.png"
        rgb = _read_rgb01(ip)
        gt = _read_mask01(gp)
        hard = thin_vessel_hard_mask(gt, fov_mask=None, cfg=cfg)

        preds = {
            "P0_zero": _pred_mask(models["P0_zero"], rgb, device, thresh),
            "J3_zero": _pred_mask(models["J3_zero"], rgb, device, thresh),
            "P0_ft": _pred_mask(models["P0_ft"], rgb, device, thresh),
            "J3_ft": _pred_mask(models["J3_ft"], rgb, device, thresh),
        }

        cols = ["RGB", "GT", "Hard(thin)", "P0(zero)", "CDC(zero)", "P0(ft)", "CDC(ft)"]
        fig, axes = plt.subplots(1, len(cols), figsize=(14, 2.4))
        axes[0].imshow(rgb)
        axes[0].set_title("RGB")
        axes[1].imshow(gt, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("GT")
        axes[2].imshow(_overlay_hard(rgb, hard))
        axes[2].set_title("Hard proxy")
        axes[3].imshow(preds["P0_zero"], cmap="gray", vmin=0, vmax=1)
        axes[3].set_title("P0 zs")
        axes[4].imshow(preds["J3_zero"], cmap="gray", vmin=0, vmax=1)
        axes[4].set_title("CDC zs")
        axes[5].imshow(preds["P0_ft"], cmap="gray", vmin=0, vmax=1)
        axes[5].set_title("P0 ft")
        axes[6].imshow(preds["J3_ft"], cmap="gray", vmin=0, vmax=1)
        axes[6].set_title("CDC ft")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        out = fig_dir / f"ext_adapt_chase_case_{idx:02d}_{cid}.png"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)

        manifest["cases"].append(
            {
                "case_id": cid,
                "file": str(out.relative_to(ROOT)),
                "image": str(ip.relative_to(ROOT)),
                "gt": str(gp.relative_to(ROOT)),
            }
        )

    (fig_dir / "ext_adapt_chase_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(cases)} panels to {fig_dir}")


if __name__ == "__main__":
    main()

