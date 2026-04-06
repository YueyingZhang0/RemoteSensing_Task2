#!/usr/bin/env python3
"""Plot J3 training dynamics + calibration ablation; write training_dynamics_summary.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def phase_spans(history: List[Dict[str, Any]]) -> List[Tuple[str, int, int]]:
    if not history:
        return []
    spans: List[Tuple[str, int, int]] = []
    start_e = int(history[0]["epoch"])
    cur = str(history[0]["phase"])
    for row in history[1:]:
        e = int(row["epoch"])
        ph = str(row["phase"])
        if ph != cur:
            spans.append((cur, start_e, e - 1))
            start_e = e
            cur = ph
    spans.append((cur, start_e, int(history[-1]["epoch"])))
    return spans


def shade_phases(ax, spans: List[Tuple[str, int, int]], colors: Dict[str, str]) -> None:
    for ph, a, b in spans:
        c = colors.get(ph, "#eeeeee")
        ax.axvspan(a - 0.5, b + 0.5, alpha=0.2, color=c, lw=0)


def load_history(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    j3_hist_path = ROOT / "outputs" / "task3_J3_joint_detach_false" / "history.json"
    cal0_path = ROOT / "outputs" / "task3_J3_calib0" / "history.json"
    cfg_path = ROOT / "outputs" / "task3_J3_joint_detach_false" / "config_used.yaml"
    fig_dir = ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    h = load_history(j3_hist_path)
    spans = phase_spans(h)
    ep = np.array([r["epoch"] for r in h], dtype=np.float64)
    colors = {
        "weight_warmup": "#a6cee3",
        "joint_calib": "#b2df8a",
        "joint_oracle_weight": "#fb9a99",
        "joint_pred_weight": "#cab2d6",
    }

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    for ax in axes:
        shade_phases(ax, spans, colors)

    axes[0].plot(ep, [r["train_seg_loss"] for r in h], "k-", lw=1.2, label="train total loss")
    td = [r.get("train_diff_loss_mean") for r in h]
    if any(x is not None for x in td):
        axes[0].plot(ep, [x if x is not None else np.nan for x in td], "g--", lw=1, label="train diff SmoothL1 (mean)")
    axes[0].set_ylabel("loss")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title("J3 joint training dynamics")

    pears = [r.get("val_sci_corr_pearson") or r.get("val_sci_corr") for r in h]
    if any(x is not None for x in pears):
        axes[1].plot(ep, [float(x) if x is not None else np.nan for x in pears], "b-", lw=1.2)
        axes[1].set_ylabel("Pearson (val)")
    else:
        axes[1].text(0.5, 0.5, "val_sci_corr_pearson not in history\n(re-train with updated task3_engine)", ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_ylabel("Pearson (val)")
    axes[1].set_title("Predicted vs GT difficulty correlation (validation)")

    axes[2].plot(ep, [r["train_weight_mean"] for r in h], label="mean", color="C0")
    axes[2].plot(ep, [r["train_weight_std"] for r in h], label="std", color="C1")
    axes[2].plot(ep, [r["train_weight_max"] for r in h], label="max", color="C2", alpha=0.7)
    axes[2].set_ylabel("sample weight")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].set_title("Training batch weight stats (oracle / pred phases)")

    axes[3].plot(ep, [r["hard_dice_mean"] for r in h], "m-", lw=1.2, label="val hard Dice")
    axes[3].plot(ep, [r["val_dice_mean"] for r in h], "c--", lw=0.8, alpha=0.7, label="val global Dice")
    axes[3].set_xlabel("epoch")
    axes[3].set_ylabel("Dice")
    axes[3].legend(loc="lower right", fontsize=8)

    handles = [plt.Rectangle((0, 0), 1, 1, fc=colors[k], alpha=0.35) for k in colors if any(s[0] == k for s in spans)]
    labels = [k for k in colors if any(s[0] == k for s in spans)]
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout()
    out_main = fig_dir / "j3_training_dynamics.png"
    fig.savefig(out_main, dpi=200, bbox_inches="tight")
    plt.close()

    # Calibration ablation
    if cal0_path.is_file():
        h0 = load_history(cal0_path)
        ep0 = np.array([r["epoch"] for r in h0], dtype=np.float64)
        fig2, ax2 = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        shade_phases(ax2[0], phase_spans(h), colors)
        shade_phases(ax2[1], phase_spans(h0), colors)
        ax2[0].plot(ep, [r["train_seg_loss"] for r in h], label="J3 (joint_calib=15)", color="C0")
        ax2[0].plot(ep0, [r["train_seg_loss"] for r in h0], label="J3_calib0", color="C1", alpha=0.85)
        ax2[0].set_ylabel("train total loss")
        ax2[0].legend()
        ax2[0].set_title("Calibration ablation: training loss")
        ax2[1].plot(ep, [r["hard_dice_mean"] for r in h], label="J3 val hard Dice", color="C0")
        ax2[1].plot(ep0, [r["hard_dice_mean"] for r in h0], label="J3_calib0", color="C1", alpha=0.85)
        ax2[1].set_xlabel("epoch")
        ax2[1].set_ylabel("val hard Dice")
        ax2[1].legend()
        plt.tight_layout()
        fig2.savefig(fig_dir / "j3_calibration_ablation_curves.png", dpi=200, bbox_inches="tight")
        plt.close()

    cfg_note = {}
    if cfg_path.is_file():
        cfg_note = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    pears_ok = any(r.get("val_sci_corr_pearson") is not None or r.get("val_sci_corr") is not None for r in h)
    summary = {
        "j3_history": str(j3_hist_path),
        "calib0_history": str(cal0_path) if cal0_path.is_file() else None,
        "phase_spans": [{"phase": a, "epoch_start": b, "epoch_end": c} for a, b, c in spans],
        "figures": {
            "j3_training_dynamics": str(out_main),
            "j3_calibration_ablation": str(fig_dir / "j3_calibration_ablation_curves.png") if cal0_path.is_file() else None,
        },
        "val_sci_corr_pearson_per_epoch_available": pears_ok,
        "config_snippet": {
            "warmup_epochs": cfg_note.get("warmup_epochs"),
            "joint_epochs": cfg_note.get("joint_epochs"),
            "difficulty_weighting": cfg_note.get("difficulty_weighting"),
        },
    }
    sum_path = ROOT / "figures" / "training_dynamics_summary.json"
    sum_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out_main}")
    print(f"Wrote {sum_path}")


if __name__ == "__main__":
    main()
