#!/usr/bin/env python3
"""Generate the final leaderboard across all experiments: P0-P6 + A1-A6 + J1-J3."""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.summarize_task3_v2_1_minimal_matrix import load_row


def load_difficulty_stats(out_dir: Path) -> dict:
    ds_path = out_dir / "difficulty_stats.json"
    if ds_path.is_file():
        return json.loads(ds_path.read_text(encoding="utf-8"))
    vm = out_dir / "val_metrics.json"
    if vm.is_file():
        v = json.loads(vm.read_text(encoding="utf-8"))
        return v.get("difficulty_stats") or {}
    return {}


ALL_RUNS = [
    ("P0_baseline", "outputs/task3_v2_1_baseline_same_split"),
    ("P2_oracle_quadratic", "outputs/task3_v2_1_oracle_quadratic_same_split"),
    ("P3_best_plus_sampler", "outputs/task3_v2_1_best_plus_sampler_same_split"),
    ("P4_frozen_probe", "outputs/task3_v3_1_predicted_frozen_same_split"),
    ("P5_oracle_matched_trigger", "outputs/task3_v3_1_oracle_matched_trigger_same_split"),
    ("P6_oracle_matched_strength", "outputs/task3_v3_1_oracle_matched_strength_same_split"),
    ("A1_tau069_a065", "outputs/task3_A1_tau069_a065"),
    ("A2_tau069_a080", "outputs/task3_A2_tau069_a080"),
    ("A3_tau071_a050", "outputs/task3_A3_tau071_a050"),
    ("A4_tau071_a065", "outputs/task3_A4_tau071_a065"),
    ("A5_tau073_a065", "outputs/task3_A5_tau073_a065"),
    ("A6_tau073_a080", "outputs/task3_A6_tau073_a080"),
    ("J1_joint_calib_only", "outputs/task3_J1_joint_calib_only"),
    ("J2_joint_detach_true", "outputs/task3_J2_joint_detach_true"),
    ("J3_joint_detach_false", "outputs/task3_J3_joint_detach_false"),
]


def main():
    rows = []
    for name, d in ALL_RUNS:
        out = ROOT / d
        if not out.is_dir():
            continue
        r = load_row(name, out)
        ds = load_difficulty_stats(out)
        r["val_sci_corr_pearson"] = ds.get("val_sci_corr_pearson") or ds.get("pearson")
        r["val_sci_corr_spearman"] = ds.get("val_sci_corr_spearman") or ds.get("spearman")
        r["pct_pred_above_tau"] = ds.get("pct_pred_above_tau") or ds.get("pct_above_tau")
        r["pct_pred_above_tau_on_true_hard"] = ds.get("pct_pred_above_tau_on_true_hard") or ds.get("pct_above_tau_on_hard")
        rows.append(r)

    fmt = lambda x: f"{x:.6f}" if x is not None else "n/a"
    fmt4 = lambda x: f"{x:.4f}" if x is not None else "n/a"
    fmt_pct = lambda x: f"{x*100:.1f}%" if x is not None else "n/a"

    print("=" * 120)
    print("  FINAL LEADERBOARD (sorted by primary_hard_dice_mean)")
    print("=" * 120)
    header = (
        f"{'Rank':<5} {'Run':<28} {'primary_hd':>12} {'hd@0.5':>10} {'vd@0.5':>10} "
        f"{'best_th':>8} {'Pearson':>8} {'Spearman':>9} {'%>tau':>7} {'%>tau_hard':>11}"
    )
    print(header)
    print("-" * 120)
    sorted_rows = sorted(rows, key=lambda x: x.get("primary_hard_dice_mean") or 0, reverse=True)
    for rank, r in enumerate(sorted_rows, 1):
        line = (
            f"{rank:<5} {r['run_id']:<28} "
            f"{fmt(r.get('primary_hard_dice_mean')):>12} "
            f"{fmt(r.get('hard_dice_mean_at_0_5')):>10} "
            f"{fmt(r.get('val_dice_mean_at_0_5')):>10} "
            f"{str(r.get('best_hard_dice_threshold','')):>8} "
            f"{fmt4(r.get('val_sci_corr_pearson')):>8} "
            f"{fmt4(r.get('val_sci_corr_spearman')):>9} "
            f"{fmt_pct(r.get('pct_pred_above_tau')):>7} "
            f"{fmt_pct(r.get('pct_pred_above_tau_on_true_hard')):>11}"
        )
        print(line)

    best_overall = sorted_rows[0]
    best_deployable_candidates = [r for r in sorted_rows if r["run_id"].startswith(("J", "P4"))]
    best_deployable = max(best_deployable_candidates, key=lambda x: x.get("primary_hard_dice_mean") or 0)

    print(f"\n{'='*80}")
    print(f"  BEST OVERALL:    {best_overall['run_id']}  primary={fmt(best_overall.get('primary_hard_dice_mean'))}")
    print(f"  BEST DEPLOYABLE: {best_deployable['run_id']}  primary={fmt(best_deployable.get('primary_hard_dice_mean'))}")
    print(f"{'='*80}")

    p0 = next(r for r in rows if r["run_id"] == "P0_baseline")
    p5 = next(r for r in rows if r["run_id"] == "P5_oracle_matched_trigger")
    p4 = next(r for r in rows if r["run_id"] == "P4_frozen_probe")

    bp = best_overall.get("primary_hard_dice_mean") or 0
    print(f"\n  Delta vs P0 baseline:              {bp - (p0.get('primary_hard_dice_mean') or 0):+.6f}")
    print(f"  Delta vs P5 oracle_matched_trigger: {bp - (p5.get('primary_hard_dice_mean') or 0):+.6f}")
    print(f"  Delta vs P4 frozen_probe:           {bp - (p4.get('primary_hard_dice_mean') or 0):+.6f}")

    out_json = ROOT / "outputs" / "final_leaderboard.json"
    out_csv = ROOT / "outputs" / "final_leaderboard.csv"

    summary = {
        "best_overall": best_overall["run_id"],
        "best_overall_primary": best_overall.get("primary_hard_dice_mean"),
        "best_deployable": best_deployable["run_id"],
        "best_deployable_primary": best_deployable.get("primary_hard_dice_mean"),
        "runs": sorted_rows,
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    keys = list(sorted_rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(sorted_rows)

    print(f"\n  Saved: {out_json}")
    print(f"  Saved: {out_csv}")


if __name__ == "__main__":
    main()
