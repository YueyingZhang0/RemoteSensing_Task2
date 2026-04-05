#!/usr/bin/env python3
"""Phase 5: Cross-split robustness analysis.

Compare J3 vs P5 vs P0 across 3 splits (original + 2 new).
Report per-split tables, aggregate summary, and deltas.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.summarize_task3_v2_1_minimal_matrix import load_row

SPLITS = {
    "splitA_rs42": {
        "P0_baseline": "outputs/task3_v2_1_baseline_same_split",
        "P5_oracle_matched_trigger": "outputs/task3_v3_1_oracle_matched_trigger_same_split",
        "J3_joint_detach_false": "outputs/task3_J3_joint_detach_false",
    },
    "splitB_rs100_seed42": {
        "P0_baseline": "outputs/task3_P0_baseline_splitB",
        "P5_oracle_matched_trigger": "outputs/task3_P5_oracle_matched_trigger_splitB",
        "J3_joint_detach_false": "outputs/task3_J3_joint_detach_false_splitB",
    },
    "splitB_rs100_seed40": {
        "P0_baseline": "outputs/task3_P0_baseline_splitB_seed40",
        "P5_oracle_matched_trigger": "outputs/task3_P5_oracle_matched_trigger_splitB_seed40",
        "J3_joint_detach_false": "outputs/task3_J3_joint_detach_false_splitB_seed40",
    },
    "splitC_rs200": {
        "P0_baseline": "outputs/task3_P0_baseline_splitC",
        "P5_oracle_matched_trigger": "outputs/task3_P5_oracle_matched_trigger_splitC",
        "J3_joint_detach_false": "outputs/task3_J3_joint_detach_false_splitC",
    },
}

PRIMARY = "primary_hard_dice_mean"
HD05 = "hard_dice_mean_at_0_5"
VD05 = "val_dice_mean_at_0_5"
BEST_TH = "best_hard_dice_threshold"
HR_BEST = "hard_recall_at_best_hard_dice_threshold"

FIELDS = [PRIMARY, HD05, VD05, BEST_TH, HR_BEST]

PREDICTOR_FIELDS = [
    "val_sci_corr_pearson",
    "val_sci_corr_spearman",
    "pct_pred_above_tau",
    "pct_pred_above_tau_on_true_hard",
]


def load_difficulty_stats(out_dir: Path) -> dict:
    ds = out_dir / "difficulty_stats.json"
    if ds.is_file():
        return json.loads(ds.read_text(encoding="utf-8"))
    return {}


def fmt(x: Any) -> str:
    if x is None:
        return "N/A"
    try:
        return f"{float(x):.6f}"
    except (TypeError, ValueError):
        return str(x)


def fmt4(x: Any) -> str:
    if x is None:
        return "N/A"
    try:
        return f"{float(x):.4f}"
    except (TypeError, ValueError):
        return str(x)


def safe_float(x: Any) -> float:
    if x is None:
        return float("nan")
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def main():
    all_rows: List[Dict[str, Any]] = []
    split_data: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for split_name, methods in SPLITS.items():
        split_data[split_name] = {}
        for method_name, out_path in methods.items():
            out_dir = ROOT / out_path
            run_id = f"{method_name}_{split_name}"
            row = load_row(run_id, out_dir)
            ds = load_difficulty_stats(out_dir)
            row["split"] = split_name
            row["method"] = method_name
            row["val_sci_corr_pearson"] = ds.get("val_sci_corr_pearson") or ds.get("pearson")
            row["val_sci_corr_spearman"] = ds.get("val_sci_corr_spearman") or ds.get("spearman")
            row["pct_pred_above_tau"] = ds.get("pct_pred_above_tau") or ds.get("pct_above_tau")
            row["pct_pred_above_tau_on_true_hard"] = ds.get("pct_pred_above_tau_on_true_hard") or ds.get("pct_above_tau_on_hard")
            all_rows.append(row)
            split_data[split_name][method_name] = row

    # =============================================
    # TABLE 1: PER-SPLIT COMPARISON
    # =============================================
    print("=" * 120)
    print("  PHASE 5: CROSS-SPLIT ROBUSTNESS ANALYSIS")
    print("=" * 120)

    print(f"\n{'='*120}")
    print("  TABLE 1: PER-SPLIT RAW METRICS")
    print(f"{'='*120}")
    header = f"{'Split':<16} {'Method':<30} {PRIMARY:>14} {'hd@0.5':>10} {'vd@0.5':>10} {'best_th':>8} {'Pearson':>8}"
    print(header)
    print("-" * 120)
    for split_name in SPLITS:
        for method_name in ["P0_baseline", "P5_oracle_matched_trigger", "J3_joint_detach_false"]:
            r = split_data[split_name][method_name]
            print(
                f"{split_name:<16} "
                f"{method_name:<30} "
                f"{fmt(r.get(PRIMARY)):>14} "
                f"{fmt(r.get(HD05)):>10} "
                f"{fmt(r.get(VD05)):>10} "
                f"{str(r.get(BEST_TH, '')):>8} "
                f"{fmt4(r.get('val_sci_corr_pearson')):>8}"
            )
        print()

    # =============================================
    # TABLE 2: PER-SPLIT DELTAS
    # =============================================
    print(f"{'='*120}")
    print("  TABLE 2: PER-SPLIT DELTAS (same-split internal comparison)")
    print(f"{'='*120}")
    print(f"{'Split':<16} {'J3-P5 primary':>15} {'J3-P0 primary':>15} {'J3-P5 hd@0.5':>15} {'J3-P0 hd@0.5':>15} {'J3-P5 vd@0.5':>15} {'J3-P0 vd@0.5':>15}")
    print("-" * 120)

    j3_wins_vs_p5 = 0
    j3_wins_vs_p0 = 0
    deltas_j3_p5 = []
    deltas_j3_p0 = []
    deltas_j3_p5_hd05 = []
    deltas_j3_p0_hd05 = []
    deltas_j3_p5_vd05 = []
    deltas_j3_p0_vd05 = []

    for split_name in SPLITS:  # noqa: C901
        j3 = split_data[split_name]["J3_joint_detach_false"]
        p5 = split_data[split_name]["P5_oracle_matched_trigger"]
        p0 = split_data[split_name]["P0_baseline"]

        d_j3_p5 = safe_float(j3.get(PRIMARY)) - safe_float(p5.get(PRIMARY))
        d_j3_p0 = safe_float(j3.get(PRIMARY)) - safe_float(p0.get(PRIMARY))
        d_j3_p5_h = safe_float(j3.get(HD05)) - safe_float(p5.get(HD05))
        d_j3_p0_h = safe_float(j3.get(HD05)) - safe_float(p0.get(HD05))
        d_j3_p5_v = safe_float(j3.get(VD05)) - safe_float(p5.get(VD05))
        d_j3_p0_v = safe_float(j3.get(VD05)) - safe_float(p0.get(VD05))

        if d_j3_p5 > 0:
            j3_wins_vs_p5 += 1
        if d_j3_p0 > 0:
            j3_wins_vs_p0 += 1

        deltas_j3_p5.append(d_j3_p5)
        deltas_j3_p0.append(d_j3_p0)
        deltas_j3_p5_hd05.append(d_j3_p5_h)
        deltas_j3_p0_hd05.append(d_j3_p0_h)
        deltas_j3_p5_vd05.append(d_j3_p5_v)
        deltas_j3_p0_vd05.append(d_j3_p0_v)

        print(
            f"{split_name:<16} "
            f"{d_j3_p5:>+15.6f} "
            f"{d_j3_p0:>+15.6f} "
            f"{d_j3_p5_h:>+15.6f} "
            f"{d_j3_p0_h:>+15.6f} "
            f"{d_j3_p5_v:>+15.6f} "
            f"{d_j3_p0_v:>+15.6f}"
        )

    n_splits = len(SPLITS)
    print(f"\n  J3 wins vs P5 on primary: {j3_wins_vs_p5}/{n_splits} entries")
    print(f"  J3 wins vs P0 on primary: {j3_wins_vs_p0}/{n_splits} entries")

    # Per-unique-split average (splitB has 2 seeds)
    print(f"\n  --- Per-unique-split aggregation (splitB averaged over 2 seeds) ---")
    unique_splits = {"splitA": ["splitA_rs42"], "splitB": ["splitB_rs100_seed42", "splitB_rs100_seed40"], "splitC": ["splitC_rs200"]}
    j3_wins_vs_p5_uniq = 0
    j3_wins_vs_p0_uniq = 0
    for uname, entries in unique_splits.items():
        avg_j3 = np.mean([safe_float(split_data[e]["J3_joint_detach_false"].get(PRIMARY)) for e in entries])
        avg_p5 = np.mean([safe_float(split_data[e]["P5_oracle_matched_trigger"].get(PRIMARY)) for e in entries])
        avg_p0 = np.mean([safe_float(split_data[e]["P0_baseline"].get(PRIMARY)) for e in entries])
        d_vs_p5 = avg_j3 - avg_p5
        d_vs_p0 = avg_j3 - avg_p0
        if d_vs_p5 > 0:
            j3_wins_vs_p5_uniq += 1
        if d_vs_p0 > 0:
            j3_wins_vs_p0_uniq += 1
        print(f"  {uname}: J3={avg_j3:.6f}, P5={avg_p5:.6f}, P0={avg_p0:.6f}  J3-P5={d_vs_p5:+.6f}  J3-P0={d_vs_p0:+.6f}")
    print(f"  J3 wins vs P5 (unique splits): {j3_wins_vs_p5_uniq}/3")
    print(f"  J3 wins vs P0 (unique splits): {j3_wins_vs_p0_uniq}/3")

    # =============================================
    # TABLE 3: AGGREGATE ACROSS SPLITS
    # =============================================
    print(f"\n{'='*120}")
    print("  TABLE 3: AGGREGATE ACROSS 3 SPLITS (mean +/- std)")
    print(f"{'='*120}")

    methods_list = ["P0_baseline", "P5_oracle_matched_trigger", "J3_joint_detach_false"]
    agg_data: Dict[str, Dict[str, Any]] = {}

    for m in methods_list:
        vals_primary = [safe_float(split_data[s][m].get(PRIMARY)) for s in SPLITS]
        vals_hd05 = [safe_float(split_data[s][m].get(HD05)) for s in SPLITS]
        vals_vd05 = [safe_float(split_data[s][m].get(VD05)) for s in SPLITS]
        agg_data[m] = {
            "primary_mean": np.nanmean(vals_primary),
            "primary_std": np.nanstd(vals_primary),
            "hd05_mean": np.nanmean(vals_hd05),
            "hd05_std": np.nanstd(vals_hd05),
            "vd05_mean": np.nanmean(vals_vd05),
            "vd05_std": np.nanstd(vals_vd05),
        }

    print(f"{'Method':<30} {'primary_hd':>22} {'hd@0.5':>22} {'vd@0.5':>22}")
    print("-" * 100)
    for m in methods_list:
        a = agg_data[m]
        print(
            f"{m:<30} "
            f"{a['primary_mean']:.6f} +/- {a['primary_std']:.6f}  "
            f"{a['hd05_mean']:.6f} +/- {a['hd05_std']:.6f}  "
            f"{a['vd05_mean']:.6f} +/- {a['vd05_std']:.6f}"
        )

    # =============================================
    # TABLE 4: MEAN DELTAS ACROSS SPLITS
    # =============================================
    print(f"\n{'='*120}")
    print("  TABLE 4: MEAN DELTAS ACROSS 3 SPLITS")
    print(f"{'='*120}")
    print(f"  J3-P5 primary_hd: mean={np.mean(deltas_j3_p5):+.6f}, std={np.std(deltas_j3_p5):.6f}")
    print(f"  J3-P0 primary_hd: mean={np.mean(deltas_j3_p0):+.6f}, std={np.std(deltas_j3_p0):.6f}")
    print(f"  J3-P5 hd@0.5:     mean={np.mean(deltas_j3_p5_hd05):+.6f}, std={np.std(deltas_j3_p5_hd05):.6f}")
    print(f"  J3-P0 hd@0.5:     mean={np.mean(deltas_j3_p0_hd05):+.6f}, std={np.std(deltas_j3_p0_hd05):.6f}")
    print(f"  J3-P5 vd@0.5:     mean={np.mean(deltas_j3_p5_vd05):+.6f}, std={np.std(deltas_j3_p5_vd05):.6f}")
    print(f"  J3-P0 vd@0.5:     mean={np.mean(deltas_j3_p0_vd05):+.6f}, std={np.std(deltas_j3_p0_vd05):.6f}")

    # =============================================
    # VERDICT
    # =============================================
    print(f"\n{'='*120}")
    print("  PASS/FAIL ASSESSMENT")
    print(f"{'='*120}")

    j3_agg = agg_data["J3_joint_detach_false"]
    p5_agg = agg_data["P5_oracle_matched_trigger"]
    p0_agg = agg_data["P0_baseline"]

    j3_gt_p5_all = j3_wins_vs_p5_uniq == 3
    j3_gt_p5_majority = j3_wins_vs_p5_uniq >= 2
    j3_gt_p0_all = j3_wins_vs_p0_uniq == 3
    j3_gt_p0_majority = j3_wins_vs_p0_uniq >= 2
    no_collapse = all(
        safe_float(split_data[s]["J3_joint_detach_false"].get(PRIMARY)) > safe_float(split_data[s]["P0_baseline"].get(PRIMARY)) - 0.005
        for s in SPLITS
    )

    print(f"\n  J3 agg primary: {j3_agg['primary_mean']:.6f} +/- {j3_agg['primary_std']:.6f}")
    print(f"  P5 agg primary: {p5_agg['primary_mean']:.6f} +/- {p5_agg['primary_std']:.6f}")
    print(f"  P0 agg primary: {p0_agg['primary_mean']:.6f} +/- {p0_agg['primary_std']:.6f}")
    print(f"\n  J3 wins vs P5: {j3_wins_vs_p5}/{n_splits} entries")
    print(f"  J3 wins vs P0: {j3_wins_vs_p0}/{n_splits} entries")
    print(f"  J3 wins vs P5 (unique splits): {j3_wins_vs_p5_uniq}/3")
    print(f"  J3 wins vs P0 (unique splits): {j3_wins_vs_p0_uniq}/3")
    print(f"  No collapse:   {no_collapse}")

    if j3_gt_p5_all and j3_gt_p0_all and no_collapse:
        verdict = "STRONG PASS: J3 >= P5 and > P0 on ALL 3 splits, no collapses"
    elif j3_gt_p5_majority and j3_gt_p0_all and no_collapse:
        verdict = "PASS: J3 >= P5 on majority of splits, > P0 on all splits, no collapses"
    elif j3_gt_p0_majority and no_collapse:
        verdict = "WEAK PASS: J3 > P0 on majority of splits, no collapses, but not always > P5"
    else:
        verdict = "NOT PASSED: J3 not consistently better than P5 and P0 across splits"

    print(f"\n  VERDICT: {verdict}")

    # =============================================
    # J3 PREDICTOR STATS ACROSS SPLITS (for J3 only)
    # =============================================
    print(f"\n{'='*120}")
    print("  J3 PREDICTOR STATISTICS ACROSS SPLITS")
    print(f"{'='*120}")
    print(f"{'Split':<16} {'Pearson':>8} {'Spearman':>8} {'%>tau':>8} {'%>tau_h':>8}")
    print("-" * 60)
    for split_name in SPLITS:
        r = split_data[split_name].get("J3_joint_detach_false", {})
        print(
            f"{split_name:<16} "
            f"{fmt4(r.get('val_sci_corr_pearson')):>8} "
            f"{fmt4(r.get('val_sci_corr_spearman')):>8} "
            f"{fmt4(r.get('pct_pred_above_tau')):>8} "
            f"{fmt4(r.get('pct_pred_above_tau_on_true_hard')):>8}"
        )

    # Save results
    out = ROOT / "outputs" / "cross_split_analysis.json"
    summary = {
        "per_split": {
            split_name: {
                method: {
                    PRIMARY: safe_float(split_data[split_name][method].get(PRIMARY)),
                    HD05: safe_float(split_data[split_name][method].get(HD05)),
                    VD05: safe_float(split_data[split_name][method].get(VD05)),
                    BEST_TH: split_data[split_name][method].get(BEST_TH),
                }
                for method in methods_list
            }
            for split_name in SPLITS
        },
        "aggregate": {m: agg_data[m] for m in methods_list},
        "deltas": {
            "j3_vs_p5_primary": deltas_j3_p5,
            "j3_vs_p0_primary": deltas_j3_p0,
            "j3_vs_p5_hd05": deltas_j3_p5_hd05,
            "j3_vs_p0_hd05": deltas_j3_p0_hd05,
            "j3_wins_vs_p5": j3_wins_vs_p5,
            "j3_wins_vs_p0": j3_wins_vs_p0,
        },
        "verdict": verdict,
    }
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    out_csv = ROOT / "outputs" / "cross_split_per_run.csv"
    keys = ["run_id", "split", "method", PRIMARY, HD05, VD05, BEST_TH, HR_BEST] + PREDICTOR_FIELDS
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)

    print(f"\n  Saved: {out}")
    print(f"  Saved: {out_csv}")


if __name__ == "__main__":
    main()
