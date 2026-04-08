#!/usr/bin/env python3
"""Figure B: HRF/all qualitative panels (thin-vessel proxy + P0/J3 zero-shot vs fine-tuned).

Outputs:
  - figures/ext_adapt_hrf_case_*.png
  - figures/ext_adapt_hrf_manifest.json
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.external.hrf_paths import hrf_case_paths  # noqa: E402
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
    patch, stride, batch_size = 128, 64, 6
    h0, w0, _ = rgb01.shape
    img = rgb01
    h, w = h0, w0
    if patch > h or patch > w:
        pad_h = max(0, patch - h)
        pad_w = max(0, patch - w)
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        h, w, _ = img.shape

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
            p = img[y : y + patch, x : x + patch, :]
            patches.append(torch.from_numpy(p).permute(2, 0, 1))
        xb = torch.stack(patches, dim=0).to(device=device, dtype=torch.float32)
        out = model(xb)
        logits = out[0] if isinstance(out, tuple) else out
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        for j, (y, x) in enumerate(bc):
            acc[y : y + patch, x : x + patch] += prob[j, 0]
            cnt[y : y + patch, x : x + patch] += 1.0

    prob_map = acc / np.maximum(cnt, 1.0)
    return (prob_map[:h0, :w0] >= float(thresh)).astype(np.uint8)


def _overlay_hard(rgb01: np.ndarray, hard: np.ndarray) -> np.ndarray:
    out = rgb01.copy()
    m = hard.astype(bool)
    out[m] = out[m] * 0.4 + np.array([1.0, 0.2, 0.2], dtype=np.float32) * 0.6
    return np.clip(out, 0, 1)


def _choose_cases(delta_csv: Path) -> List[str]:
    df = pd.read_csv(delta_csv)
    best = df.sort_values("delta", ascending=False).iloc[0]["case_id"]
    mid = df.sort_values("delta", ascending=False).iloc[len(df) // 2]["case_id"]
    worst = df.sort_values("delta", ascending=True).iloc[0]["case_id"]
    out: List[str] = []
    for c in [best, mid, worst]:
        s = str(c)
        if s not in out:
            out.append(s)
    return out


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    thresh = 0.5

    p0_csv = ROOT / "outputs" / "ft_hrf_p0_seed42" / "external_test_per_image_metrics.csv"
    j3_csv = ROOT / "outputs" / "ft_hrf_j3_seed42" / "external_test_per_image_metrics.csv"
    p0 = pd.read_csv(p0_csv)[["case_id", "dice@0.5"]].rename(columns={"dice@0.5": "p0"})
    j3 = pd.read_csv(j3_csv)[["case_id", "dice@0.5"]].rename(columns={"dice@0.5": "j3"})
    m = p0.merge(j3, on="case_id", how="inner")
    m["delta"] = m["j3"] - m["p0"]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as tf:
        m.to_csv(tf.name, index=False)
        tmp = Path(tf.name)
    try:
        cases = _choose_cases(tmp)
    finally:
        tmp.unlink(missing_ok=True)

    models = {
        "P0_zero": _load_model("baseline", ROOT / "outputs" / "task3_v2_1_baseline_same_split" / "best_model.pt", device),
        "J3_zero": _load_model("joint", ROOT / "outputs" / "task3_J3_joint_detach_false" / "best_model.pt", device),
        "P0_ft": _load_model("baseline", ROOT / "outputs" / "ft_hrf_p0_seed42" / "best_model.pt", device),
        "J3_ft": _load_model("joint", ROOT / "outputs" / "ft_hrf_j3_seed42" / "best_model.pt", device),
    }

    cfg = ThinVesselProxyConfig(r_th=2, dilate_iters=1)
    fig_dir = ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {"dataset": "HRF all (manual1; FOV mask)", "threshold": thresh, "cases": []}

    for idx, cid in enumerate(cases, start=1):
        paths = hrf_case_paths(cid, ROOT)
        rgb = _read_rgb01(paths["image"])
        gt = _read_mask01(paths["gt"])
        fov = _read_mask01(paths["fov"])
        hard = thin_vessel_hard_mask(gt, fov_mask=fov, cfg=cfg)

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
        safe = str(cid).replace("/", "_")
        out = fig_dir / f"ext_adapt_hrf_case_{idx:02d}_{safe}.png"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)

        manifest["cases"].append(
            {
                "case_id": cid,
                "file": str(out.relative_to(ROOT)),
                "image": str(paths["image"].relative_to(ROOT)),
                "gt": str(paths["gt"].relative_to(ROOT)),
                "fov": str(paths["fov"].relative_to(ROOT)),
            }
        )

    (fig_dir / "ext_adapt_hrf_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(cases)} HRF panels to {fig_dir}")


if __name__ == "__main__":
    main()
