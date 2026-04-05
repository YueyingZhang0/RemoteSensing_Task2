#!/usr/bin/env python3
"""Unified summary: P0/P2/P3/P4 frozen probe + P5/P6 matched-sparsity oracle (same primary metric as v2.1)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# Reuse row loader from v2.1 summarizer
from src.analysis.summarize_task3_v2_1_minimal_matrix import load_row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=".")
    args = p.parse_args()
    root = Path(args.root).resolve()

    runs: List[tuple[str, Path]] = [
        ("P0_baseline", root / "outputs" / "task3_v2_1_baseline_same_split"),
        ("P2_oracle_quadratic", root / "outputs" / "task3_v2_1_oracle_quadratic_same_split"),
        ("P3_best_plus_sampler", root / "outputs" / "task3_v2_1_best_plus_sampler_same_split"),
        ("P4_frozen_probe", root / "outputs" / "task3_v3_1_predicted_frozen_same_split"),
        ("P5_oracle_matched_trigger", root / "outputs" / "task3_v3_1_oracle_matched_trigger_same_split"),
        ("P6_oracle_matched_strength", root / "outputs" / "task3_v3_1_oracle_matched_strength_same_split"),
    ]
    rows = [load_row(rid, d) for rid, d in runs]

    p4 = next(r for r in rows if r["run_id"] == "P4_frozen_probe")
    p0 = rows[0]
    p2 = rows[1]

    summary: Dict[str, Any] = {
        "runs": rows,
        "primary_metric": "hard_dice_mean_at_best_hard_dice_threshold",
        "frozen_probe_vs_baseline": {
            "delta_primary_hard_dice_vs_P0": _fdiff(p4, p0, "primary_hard_dice_mean"),
            "delta_hard_dice_at_0_5_vs_P0": _fdiff(p4, p0, "hard_dice_mean_at_0_5"),
            "delta_val_dice_at_0_5_vs_P0": _fdiff(p4, p0, "val_dice_mean_at_0_5"),
        },
        "frozen_probe_vs_oracle_quadratic": {
            "delta_primary_hard_dice_vs_P2": _fdiff(p4, p2, "primary_hard_dice_mean"),
        },
        "interpretation_hints": {
            "fixed_at_0_5_check": "Compare hard_dice_mean_at_0_5 and val_dice_mean_at_0_5 across runs",
            "primary_check": "Compare primary_hard_dice_mean (best hard-dice threshold on val)",
            "matched_controls": "If P5/P6 ~= P4 on primary, sparse/regularization story; if P4 >> P5/P6, prediction-specific story",
        },
    }

    out_dir = root / "outputs" / "task3_v3_1_predicted_frozen_same_split"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_j = out_dir / "frozen_probe_summary.json"
    out_c = out_dir / "frozen_probe_summary.csv"
    out_j.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    keys = list(rows[0].keys())
    with open(out_c, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    print("run_id".ljust(28), "primary_hd".ljust(14), "hd@0.5".ljust(12), "val@0.5".ljust(12), "best_hd_th")
    print("-" * 80)
    for r in rows:
        print(
            str(r["run_id"]).ljust(28),
            _fmt(r.get("primary_hard_dice_mean")).ljust(14),
            _fmt(r.get("hard_dice_mean_at_0_5")).ljust(12),
            _fmt(r.get("val_dice_mean_at_0_5")).ljust(12),
            str(r.get("best_hard_dice_threshold")),
        )
    print(f"\nWrote {out_j}")
    print(f"Wrote {out_c}")


def _fmt(x: Any) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{float(x):.6f}"
    except (TypeError, ValueError):
        return str(x)


def _fdiff(a: Dict[str, Any], b: Dict[str, Any], key: str) -> Optional[float]:
    try:
        va = float(a[key])
        vb = float(b[key])
        if va != va or vb != vb:
            return None
        return va - vb
    except (KeyError, TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
