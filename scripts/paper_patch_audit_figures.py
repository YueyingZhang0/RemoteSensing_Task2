#!/usr/bin/env python3
"""Scatter + example thumbnails from analysis/j3_patch_audit.csv."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    csv_path = ROOT / "analysis" / "j3_patch_audit.csv"
    if not csv_path.is_file():
        raise SystemExit(f"Missing {csv_path}; run: python -m src.analysis.j3_patch_audit")

    audit = pd.read_csv(csv_path)
    fig_dir = ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    x = audit["gt_sci_res2_norm"].values
    y = audit["delta_dice_j3_vs_p0"].values
    hard = audit["is_hard"].values.astype(bool)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(x[~hard], y[~hard], c="#888888", s=18, alpha=0.6, label="not hard")
    ax.scatter(x[hard], y[hard], c="#c51b7d", s=28, alpha=0.75, label="hard (val)")
    ax.axhline(0, color="k", lw=0.8, linestyle="--")
    ax.set_xlabel("GT difficulty (sci_res2_norm)")
    ax.set_ylabel("Δ Dice (J3 − P0)")
    ax.set_title("Patch-level gain vs difficulty")
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    p_sc = fig_dir / "j3_patch_scatter.png"
    fig.savefig(p_sc, dpi=200)
    plt.close()
    print(f"Wrote {p_sc}")

    img_dir = ROOT / "outputs" / "task1_topo_v3" / "patches" / "images"
    top = audit.nlargest(12, "delta_dice_j3_vs_p0")
    fig2, axes = plt.subplots(3, 4, figsize=(10, 7))
    axes = axes.ravel()
    for i, (_, row) in enumerate(top.iterrows()):
        if i >= 12:
            break
        pn = row["patch_name"]
        ip = img_dir / pn
        if ip.is_file():
            im = plt.imread(ip)
            axes[i].imshow(np.clip(im, 0, 1))
        axes[i].set_title(f"{pn[:18]}…\nΔ={row['delta_dice_j3_vs_p0']:.3f}", fontsize=7)
        axes[i].axis("off")
    for j in range(len(top), 12):
        axes[j].axis("off")
    plt.suptitle("Top J3 vs P0 gains (RGB patches)", fontsize=11)
    plt.tight_layout()
    p_ex = fig_dir / "j3_patch_examples_panel.png"
    fig2.savefig(p_ex, dpi=160)
    plt.close()
    print(f"Wrote {p_ex}")


if __name__ == "__main__":
    main()
