"""
Summarise cross-split hard-Dice results for P0, P5, and J3.

Reads outputs/cross_split_analysis.json and reports primary_hard_dice_mean
across the actually available split × seed combinations.  Does NOT assume
a fixed number of combinations.

Output:
    outputs/revision_analysis/cross_split_summary.json
    outputs/revision_analysis/cross_split_summary.csv
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT     = Path(__file__).resolve().parents[2]
SRC_JSON = ROOT / "outputs" / "cross_split_analysis.json"
OUT_DIR  = ROOT / "outputs" / "revision_analysis"

# Canonical condition keys as found in the JSON
CONDITIONS = [
    "P0_baseline",
    "P5_oracle_matched_trigger",
    "J3_joint_detach_false",
]

FRIENDLY = {
    "P0_baseline":               "P0",
    "P5_oracle_matched_trigger": "P5",
    "J3_joint_detach_false":     "J3",
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load JSON
    # ------------------------------------------------------------------
    if not SRC_JSON.exists():
        print(f"[ERROR] Not found: {SRC_JSON}", file=sys.stderr)
        sys.exit(1)

    with SRC_JSON.open() as fh:
        data = json.load(fh)

    per_split  = data.get("per_split", {})
    aggregate  = data.get("aggregate", {})
    splits_found = list(per_split.keys())

    if not splits_found:
        print("[ERROR] 'per_split' section is empty.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(splits_found)} split × seed combination(s): {splits_found}")

    # ------------------------------------------------------------------
    # Per-split table
    # ------------------------------------------------------------------
    split_records = []
    for split_id, split_data in per_split.items():
        row: dict = {"split": split_id}
        for cond in CONDITIONS:
            friendly = FRIENDLY.get(cond, cond)
            cond_data = split_data.get(cond)
            if cond_data is None:
                warnings.warn(
                    f"[MISSING] Condition '{cond}' not found in split '{split_id}'.",
                    UserWarning,
                    stacklevel=1,
                )
                row[f"{friendly}_primary_hard_dice"] = float("nan")
            else:
                val = cond_data.get("primary_hard_dice_mean")
                if val is None:
                    warnings.warn(
                        f"[MISSING KEY] 'primary_hard_dice_mean' missing for "
                        f"condition '{cond}' in split '{split_id}'.",
                        UserWarning,
                        stacklevel=1,
                    )
                row[f"{friendly}_primary_hard_dice"] = val
        split_records.append(row)

    split_df = pd.DataFrame(split_records)

    # ------------------------------------------------------------------
    # Aggregate summary (computed from per_split values actually present)
    # ------------------------------------------------------------------
    agg_records = []
    for cond in CONDITIONS:
        friendly = FRIENDLY.get(cond, cond)
        col = f"{friendly}_primary_hard_dice"
        vals = split_df[col].dropna().values

        # Also check whether the JSON's own pre-computed aggregate matches
        json_agg = aggregate.get(cond, {})
        json_mean = json_agg.get("primary_mean")
        json_std  = json_agg.get("primary_std")

        computed_mean = float(np.mean(vals)) if len(vals) > 0 else float("nan")
        computed_std  = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")

        if json_mean is not None and not np.isnan(computed_mean):
            if abs(computed_mean - json_mean) > 1e-6:
                warnings.warn(
                    f"[MISMATCH] {cond}: JSON aggregate mean {json_mean:.6f} "
                    f"differs from recomputed mean {computed_mean:.6f}.",
                    UserWarning,
                    stacklevel=1,
                )

        agg_records.append({
            "condition":          cond,
            "friendly_name":      friendly,
            "n_combinations":     int(len(vals)),
            "primary_hard_dice_mean": computed_mean,
            "primary_hard_dice_std":  computed_std,
            "json_aggregate_mean":    json_mean,
            "json_aggregate_std":     json_std,
        })

    agg_df = pd.DataFrame(agg_records)

    # ------------------------------------------------------------------
    # Print
    # ------------------------------------------------------------------
    print("\n=== Per split × seed ===")
    print(split_df.to_string(index=False))
    print("\n=== Aggregate (computed from available combinations) ===")
    print(agg_df.to_string(index=False))

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    # CSV — per-split and aggregate stacked
    split_df_out  = split_df.copy()
    split_df_out["table"] = "per_split"
    agg_df_out            = agg_df.copy()
    agg_df_out["table"]   = "aggregate"

    csv_path = OUT_DIR / "cross_split_summary.csv"
    split_df_out.to_csv(csv_path, index=False)
    print(f"\nSaved CSV  → {csv_path}")

    # JSON — combined structure
    out_json = {
        "n_combinations": len(splits_found),
        "splits": splits_found,
        "per_split": split_df.to_dict(orient="records"),
        "aggregate": agg_df.to_dict(orient="records"),
        "verdict": data.get("verdict", ""),
    }
    json_path = OUT_DIR / "cross_split_summary.json"
    with json_path.open("w") as fh:
        json.dump(out_json, fh, indent=2)
    print(f"Saved JSON → {json_path}")


if __name__ == "__main__":
    main()
