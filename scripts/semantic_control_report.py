#!/usr/bin/env python3
"""Phase 6: Compare J3 semantic control (lambda_diff=0) vs J3 vs P0."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.summarize_task3_v2_1_minimal_matrix import load_row

RUNS = {
    "P0_baseline": ROOT / "outputs/task3_v2_1_baseline_same_split",
    "J3_joint_detach_false": ROOT / "outputs/task3_J3_joint_detach_false",
    "J3_semantic_control": ROOT / "outputs/task3_J3_semantic_control",
}

PREDICTOR_KEYS = [
    "val_sci_corr_pearson", "pearson",
    "val_sci_corr_spearman", "spearman",
    "pct_pred_above_tau", "pct_above_tau",
    "pct_pred_above_tau_on_true_hard", "pct_above_tau_on_hard",
    "pred_mean", "pred_std", "mean", "std",
]


def fmt(x):
    if x is None:
        return "N/A"
    try:
        return f"{float(x):.6f}"
    except (TypeError, ValueError):
        return str(x)


def main():
    print("=" * 100)
    print("  PHASE 6: SEMANTIC CONTROL COMPARISON (same split, seed=42)")
    print("=" * 100)

    header = f"{'Run':<30} {'primary_hd':>14} {'hd@0.5':>10} {'vd@0.5':>10} {'best_th':>8}"
    print(header)
    print("-" * 80)

    rows = {}
    for name, d in RUNS.items():
        r = load_row(name, d)
        rows[name] = r
        phd = r.get("primary_hard_dice_mean")
        hd = r.get("hard_dice_mean_at_0_5")
        vd = r.get("val_dice_mean_at_0_5")
        bt = r.get("best_hard_dice_threshold")
        print(f"{name:<30} {fmt(phd):>14} {fmt(hd):>10} {fmt(vd):>10} {str(bt):>8}")

    for label in ["J3_joint_detach_false", "J3_semantic_control"]:
        ds_path = ROOT / f"outputs/task3_{label.replace('J3_', 'J3_')}/difficulty_stats.json"
        if not ds_path.is_file():
            ds_path = ROOT / f"outputs/task3_{label}/difficulty_stats.json"
        if ds_path.is_file():
            ds = json.loads(ds_path.read_text(encoding="utf-8"))
            print(f"\n  {label} predictor stats:")
            for k in PREDICTOR_KEYS:
                if k in ds:
                    print(f"    {k}: {ds[k]}")

    j3_phd = float(rows["J3_joint_detach_false"].get("primary_hard_dice_mean") or 0)
    sc_phd = float(rows["J3_semantic_control"].get("primary_hard_dice_mean") or 0)
    p0_phd = float(rows["P0_baseline"].get("primary_hard_dice_mean") or 0)

    j3_hd05 = float(rows["J3_joint_detach_false"].get("hard_dice_mean_at_0_5") or 0)
    sc_hd05 = float(rows["J3_semantic_control"].get("hard_dice_mean_at_0_5") or 0)
    p0_hd05 = float(rows["P0_baseline"].get("hard_dice_mean_at_0_5") or 0)

    print(f"\n{'='*80}")
    print("  DELTAS")
    print(f"{'='*80}")
    print(f"  J3 primary_hd:                {j3_phd:.6f}")
    print(f"  Semantic control primary_hd:  {sc_phd:.6f}")
    print(f"  P0 baseline primary_hd:       {p0_phd:.6f}")
    print(f"  J3 - semantic control:        {j3_phd - sc_phd:+.6f}")
    print(f"  Semantic control - P0:        {sc_phd - p0_phd:+.6f}")
    print()
    print(f"  J3 hd@0.5:                    {j3_hd05:.6f}")
    print(f"  Semantic control hd@0.5:      {sc_hd05:.6f}")
    print(f"  P0 baseline hd@0.5:           {p0_hd05:.6f}")
    print(f"  J3 - semantic control @0.5:   {j3_hd05 - sc_hd05:+.6f}")
    print(f"  Semantic control - P0 @0.5:   {sc_hd05 - p0_hd05:+.6f}")

    print(f"\n{'='*80}")
    print("  CONCLUSION")
    print(f"{'='*80}")
    if sc_phd < p0_phd - 0.001:
        print("  Semantic control WORSE than baseline.")
        print("  => J3 gains are NOT from architecture/noise, but from meaningful difficulty signal.")
        conclusion = "PASS: difficulty signal is meaningful"
    elif abs(sc_phd - p0_phd) <= 0.001:
        print("  Semantic control CLOSE to baseline.")
        print("  => Random predicted weights do not help. J3 gain requires trained difficulty signal.")
        conclusion = "PASS: difficulty signal is meaningful (control matches baseline)"
    elif abs(sc_phd - j3_phd) < 0.001:
        print("  Semantic control CLOSE to J3.")
        print("  => J3 gains may come from architecture, not difficulty signal. CONCERNING.")
        conclusion = "FAIL: gain may be from architecture, not difficulty signal"
    else:
        print("  Semantic control between baseline and J3.")
        print("  => Part of J3 gain comes from architecture, part from difficulty signal.")
        conclusion = "PARTIAL: some gain from architecture, some from signal"

    print(f"\n  VERDICT: {conclusion}")

    result = {
        "J3_primary_hd": j3_phd,
        "semantic_control_primary_hd": sc_phd,
        "P0_primary_hd": p0_phd,
        "delta_J3_minus_SC": j3_phd - sc_phd,
        "delta_SC_minus_P0": sc_phd - p0_phd,
        "conclusion": conclusion,
    }
    out = ROOT / "outputs" / "semantic_control_result.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
