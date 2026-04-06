#!/usr/bin/env python3
"""tables/cross_split_summary.{csv,md} from outputs/cross_split_analysis.json."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    src = ROOT / "outputs" / "cross_split_analysis.json"
    data = json.loads(src.read_text(encoding="utf-8"))
    per = data.get("per_split", {})
    rows = []
    for split_name, methods in per.items():
        for method, m in methods.items():
            rows.append(
                {
                    "split": split_name,
                    "method": method,
                    "primary_hd": m.get("primary_hard_dice_mean"),
                    "hd_0.5": m.get("hard_dice_mean_at_0_5"),
                    "vd_0.5": m.get("val_dice_mean_at_0_5"),
                    "best_hard_th": m.get("best_hard_dice_threshold"),
                }
            )

    out_dir = ROOT / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = ["split", "method", "primary_hd", "hd_0.5", "vd_0.5", "best_hard_th"]
    csv_path = out_dir / "cross_split_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    lines = [
        "## Cross-split summary",
        "",
        "Interpret within-split deltas; absolute primary across splits not comparable (hard subset changes).",
        "",
        "| " + " | ".join(keys) + " |",
        "| " + " | ".join(["---"] * len(keys)) + " |",
    ]
    for r in rows:
        lines.append("| " + " | ".join(str(r[k]) for k in keys) + " |")
    (out_dir / "cross_split_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
