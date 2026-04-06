#!/usr/bin/env python3
"""Build tables/main_results_final.{csv,md} for P0 / P5 / J3 (canonical + multi-seed + cross-split)."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.summarize_task3_v2_1_minimal_matrix import load_row

CANONICAL = {
    "P0_baseline": ROOT / "outputs" / "task3_v2_1_baseline_same_split",
    "P5_oracle_matched_trigger": ROOT / "outputs" / "task3_v3_1_oracle_matched_trigger_same_split",
    "J3_joint_detach_false": ROOT / "outputs" / "task3_J3_joint_detach_false",
}


def main() -> None:
    multiseed_path = ROOT / "outputs" / "multiseed_analysis.json"
    cross_path = ROOT / "outputs" / "cross_split_analysis.json"
    multiseed = json.loads(multiseed_path.read_text(encoding="utf-8")) if multiseed_path.is_file() else {}
    cross = json.loads(cross_path.read_text(encoding="utf-8")) if cross_path.is_file() else {}

    agg_ms = {a["method"]: a for a in multiseed.get("aggregate", [])}
    agg_x = cross.get("aggregate", {})

    rows = []
    for method, odir in CANONICAL.items():
        r = load_row(method, odir)
        ms = agg_ms.get(method, {})
        xs = agg_x.get(method, {})
        rows.append(
            {
                "method": method,
                "primary_hd_seed42": r.get("primary_hard_dice_mean"),
                "hd_at_0_5_seed42": r.get("hard_dice_mean_at_0_5"),
                "vd_at_0_5_seed42": r.get("val_dice_mean_at_0_5"),
                "best_hard_dice_th_seed42": r.get("best_hard_dice_threshold"),
                "hard_recall_at_best_hd_seed42": r.get("hard_recall_at_best_hard_dice_threshold"),
                "multi_seed_primary_hd_mean": ms.get("primary_hard_dice_mean_mean"),
                "multi_seed_primary_hd_std": ms.get("primary_hard_dice_mean_std"),
                "multi_seed_hd_0_5_mean": ms.get("hard_dice_mean_at_0_5_mean"),
                "multi_seed_hd_0_5_std": ms.get("hard_dice_mean_at_0_5_std"),
                "multi_seed_vd_0_5_mean": ms.get("val_dice_mean_at_0_5_mean"),
                "multi_seed_vd_0_5_std": ms.get("val_dice_mean_at_0_5_std"),
                "multi_seed_best_hd_th_mean": ms.get("best_hard_dice_threshold_mean"),
                "multi_seed_hr_at_best_hd_mean": ms.get("hard_recall_at_best_hard_dice_threshold_mean"),
                "multi_seed_hr_at_best_hd_std": ms.get("hard_recall_at_best_hard_dice_threshold_std"),
                "cross_split_primary_hd_mean": xs.get("primary_mean"),
                "cross_split_primary_hd_std": xs.get("primary_std"),
                "cross_split_hd_0_5_mean": xs.get("hd05_mean"),
                "cross_split_hd_0_5_std": xs.get("hd05_std"),
                "cross_split_vd_0_5_mean": xs.get("vd05_mean"),
                "cross_split_vd_0_5_std": xs.get("vd05_std"),
            }
        )

    out_dir = ROOT / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "main_results_final.csv"
    md_path = out_dir / "main_results_final.md"

    keys = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    note = """
## Main results (Task 3, frozen protocol)

**Primary endpoint:** `hard_dice_mean` at **best_hard_dice_threshold** (selected on validation grid; same grid for all methods).

**Secondary endpoints:** `hard_dice_mean` at fixed threshold 0.5; `val_dice_mean` at 0.5.

**Canonical split:** `outputs/task3_split/split_info.json`, `random_state=42` (seed42 columns).

**Multi-seed:** mean ± std over seeds 40, 41, 42 on the **same** split (see `outputs/multiseed_analysis.json`).

**Cross-split:** mean ± std over run entries in `outputs/cross_split_analysis.json` (includes alternate splits and an extra seed on split B). Absolute primary values are **not** comparable across splits (hard subset changes); use within-split deltas for robustness claims. Aggregate means are reported for summary only.

"""
    lines = [note.strip(), "", "| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows:
        cells = []
        for k in keys:
            v = row[k]
            if v is None:
                cells.append("")
            elif isinstance(v, float):
                cells.append(f"{v:.6f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
