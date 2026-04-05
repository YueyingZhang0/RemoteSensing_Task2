#!/usr/bin/env python3
"""Phase 4: Warmup ablation report - joint_calib_epochs=0 vs 15."""
import json
import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.summarize_task3_v2_1_minimal_matrix import load_row

RUNS = [
    ("J3_calib15 (reference)", "outputs/task3_J3_joint_detach_false", 15),
    ("J3_calib0", "outputs/task3_J3_calib0", 0),
]

fmt = lambda x: f"{x:.6f}" if x is not None else "N/A"
fmt4 = lambda x: f"{x:.4f}" if x is not None else "N/A"
fmt_pct = lambda x: f"{x*100:.1f}%" if x is not None else "N/A"


def load_ds(d: Path) -> dict:
    ds = d / "difficulty_stats.json"
    return json.loads(ds.read_text(encoding="utf-8")) if ds.is_file() else {}


def main():
    print("=" * 100)
    print("  PHASE 4: WARMUP ABLATION (joint_calib_epochs)")
    print("=" * 100)

    rows = []
    for name, out_path, calib_ep in RUNS:
        out_dir = ROOT / out_path
        r = load_row(name, out_dir)
        ds = load_ds(out_dir)
        r["calib_epochs"] = calib_ep
        r["val_sci_corr_pearson"] = ds.get("val_sci_corr_pearson") or ds.get("pearson")
        r["val_sci_corr_spearman"] = ds.get("val_sci_corr_spearman") or ds.get("spearman")
        r["pct_pred_above_tau"] = ds.get("pct_pred_above_tau") or ds.get("pct_above_tau")
        r["pct_pred_above_tau_on_true_hard"] = ds.get("pct_pred_above_tau_on_true_hard") or ds.get("pct_above_tau_on_hard")
        rows.append(r)

    header = (
        f"{'Run':<30} {'calib':>6} {'primary_hd':>12} {'hd@0.5':>10} {'vd@0.5':>10} "
        f"{'best_th':>8} {'Pearson':>8} {'%>tau_h':>8}"
    )
    print(header)
    print("-" * 100)
    for r in rows:
        print(
            f"{r['run_id']:<30} "
            f"{r['calib_epochs']:>6} "
            f"{fmt(r.get('primary_hard_dice_mean')):>12} "
            f"{fmt(r.get('hard_dice_mean_at_0_5')):>10} "
            f"{fmt(r.get('val_dice_mean_at_0_5')):>10} "
            f"{str(r.get('best_hard_dice_threshold','')):>8} "
            f"{fmt4(r.get('val_sci_corr_pearson')):>8} "
            f"{fmt_pct(r.get('pct_pred_above_tau_on_true_hard')):>8}"
        )

    ref = rows[0]
    abl = rows[1]
    delta = (abl.get("primary_hard_dice_mean") or 0) - (ref.get("primary_hard_dice_mean") or 0)
    print(f"\nDelta (calib0 - calib15) primary: {delta:+.6f}")

    if delta < -0.003:
        print("CONCLUSION: Calibration period is IMPORTANT. Removing it causes significant drop.")
    elif delta < -0.001:
        print("CONCLUSION: Calibration period provides a modest benefit.")
    elif abs(delta) < 0.001:
        print("CONCLUSION: Calibration period has negligible effect.")
    else:
        print("CONCLUSION: Removing calibration period actually helps slightly.")

    out = ROOT / "outputs" / "warmup_ablation.json"
    summary = {"runs": rows, "delta_calib0_minus_calib15": delta}
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
