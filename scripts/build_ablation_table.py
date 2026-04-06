#!/usr/bin/env python3
"""tables/ablation_mechanism.{csv,md}: J1/J2/J3 + calib0 + semantic control."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.summarize_task3_v2_1_minimal_matrix import load_row

RUNS = [
    ("J1_joint_calib_only", ROOT / "outputs" / "task3_J1_joint_calib_only"),
    ("J2_joint_detach_true", ROOT / "outputs" / "task3_J2_joint_detach_true"),
    ("J3_joint_detach_false", ROOT / "outputs" / "task3_J3_joint_detach_false"),
    ("J3_calib0", ROOT / "outputs" / "task3_J3_calib0"),
    ("J3_semantic_lambda_diff_0", ROOT / "outputs" / "task3_J3_semantic_control"),
]


def main() -> None:
    rows = []
    for name, odir in RUNS:
        if not (odir / "threshold_metrics.json").is_file():
            rows.append({"run": name, "note": "missing artifacts", "primary_hd": "", "hd_0.5": "", "vd_0.5": "", "pearson": ""})
            continue
        r = load_row(name, odir)
        ds_path = odir / "difficulty_stats.json"
        pear = ""
        if ds_path.is_file():
            ds = json.loads(ds_path.read_text(encoding="utf-8"))
            pear = ds.get("val_sci_corr_pearson") or ds.get("pearson")
            if pear is not None:
                pear = f"{float(pear):.4f}"
        rows.append(
            {
                "run": name,
                "primary_hd": r.get("primary_hard_dice_mean"),
                "hd_0.5": r.get("hard_dice_mean_at_0_5"),
                "vd_0.5": r.get("val_dice_mean_at_0_5"),
                "best_hard_th": r.get("best_hard_dice_threshold"),
                "pearson_val": pear,
                "note": "",
            }
        )

    out_dir = ROOT / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = ["run", "primary_hd", "hd_0.5", "vd_0.5", "best_hard_th", "pearson_val", "note"]
    csv_path = out_dir / "ablation_mechanism.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in keys})

    md = ["## Mechanism ablations (same split as main, seed 42 unless noted)", "", "| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows:
        md.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    (out_dir / "ablation_mechanism.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
