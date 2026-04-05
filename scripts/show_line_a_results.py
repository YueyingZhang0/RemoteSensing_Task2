#!/usr/bin/env python3
"""Show Line A results table using the same load_row logic as v2.1 summarizer."""
import json
import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from src.analysis.summarize_task3_v2_1_minimal_matrix import load_row

runs = [
    ("P5_reference", root / "outputs" / "task3_v3_1_oracle_matched_trigger_same_split", 0.7076, 0.80),
    ("A1_tau069_a065", root / "outputs" / "task3_A1_tau069_a065", 0.69, 0.65),
    ("A2_tau069_a080", root / "outputs" / "task3_A2_tau069_a080", 0.69, 0.80),
    ("A3_tau071_a050", root / "outputs" / "task3_A3_tau071_a050", 0.71, 0.50),
    ("A4_tau071_a065", root / "outputs" / "task3_A4_tau071_a065", 0.71, 0.65),
    ("A5_tau073_a065", root / "outputs" / "task3_A5_tau073_a065", 0.73, 0.65),
    ("A6_tau073_a080", root / "outputs" / "task3_A6_tau073_a080", 0.73, 0.80),
]

rows = []
for name, d, tau, alpha in runs:
    r = load_row(name, d)
    r["tau"] = tau
    r["alpha"] = alpha
    rows.append(r)

fmt = lambda x: f"{x:.6f}" if x is not None else "n/a"

header = f"{'run':<22} {'tau':>6} {'alpha':>6} {'primary_hd':>12} {'hd@0.5':>10} {'vd@0.5':>10} {'best_th':>8}"
print(header)
print("-" * len(header))
for r in sorted(rows, key=lambda x: x.get("primary_hard_dice_mean") or 0, reverse=True):
    print(
        f"{r['run_id']:<22} "
        f"{r['tau']:>6} "
        f"{r['alpha']:>6} "
        f"{fmt(r.get('primary_hard_dice_mean')):>12} "
        f"{fmt(r.get('hard_dice_mean_at_0_5')):>10} "
        f"{fmt(r.get('val_dice_mean_at_0_5')):>10} "
        f"{str(r.get('best_hard_dice_threshold', '')):>8}"
    )

best_a = max(
    [r for r in rows if r["run_id"] != "P5_reference"],
    key=lambda x: x.get("primary_hard_dice_mean") or 0,
)
ref = 0.7374339064055633
bp = best_a.get("primary_hard_dice_mean") or 0
print(f"\nBest Line A new: {best_a['run_id']}  primary={fmt(best_a.get('primary_hard_dice_mean'))}")
print(f"P5 reference:                          primary={ref:.6f}")
print(f"Delta vs P5: {bp - ref:+.6f}")
if bp >= 0.7380:
    print(">>> STOP CONDITION MET: primary >= 0.7380 <<<")
else:
    print(f"Stop condition NOT met (need >= 0.7380, got {bp:.6f})")

out_json = root / "outputs" / "line_a_summary.json"
out_json.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(f"\nSaved: {out_json}")
