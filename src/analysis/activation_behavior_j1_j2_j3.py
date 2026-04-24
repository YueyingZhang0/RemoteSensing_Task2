"""
Activation-behaviour comparison across J1, J2, and J3 runs.

For each run, extracts from difficulty_stats.json:
  - Pearson r        (val_sci_corr_pearson)
  - Spearman rho     (val_sci_corr_spearman)   -- present in all three runs
  - A(tau)           (pct_pred_above_tau)       -- fraction of patches where
                                                   predicted SCI > tau
  - A_hard(tau)      (pct_pred_above_tau_on_true_hard) -- same for true-hard patches
  - n_hard           (n_hard)
  - tau              (tau)

Also reads hard_patch_metrics.json for hard_dice_mean.

J1 activation statistics are annotated as COUNTERFACTUAL because the joint
dynamic-weighting loss is NOT active during J1 training (J1 uses calibration
only, no gradient-based weighting).

Output: outputs/revision_analysis/activation_behavior_j1_j2_j3.csv
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]

RUNS: dict[str, dict] = {
    "J1": {
        "dir": ROOT / "outputs" / "task3_J1_joint_calib_only",
        "note": "COUNTERFACTUAL — dynamic weighting not active during training",
    },
    "J2": {
        "dir": ROOT / "outputs" / "task3_J2_joint_detach_true",
        "note": "",
    },
    "J3": {
        "dir": ROOT / "outputs" / "task3_J3_joint_detach_false",
        "note": "",
    },
}

OUT_DIR = ROOT / "outputs" / "revision_analysis"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict | None:
    if not path.exists():
        warnings.warn(f"[MISSING] {path}", UserWarning, stacklevel=2)
        return None
    with path.open() as fh:
        return json.load(fh)


def safe_get(d: dict | None, key: str, label: str) -> float | None:
    if d is None:
        return None
    if key not in d:
        warnings.warn(
            f"[MISSING KEY] '{key}' not found in {label}", UserWarning, stacklevel=2
        )
        return None
    return d[key]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    for run_name, meta in RUNS.items():
        run_dir = meta["dir"]

        if not run_dir.exists():
            warnings.warn(
                f"[MISSING RUN DIR] {run_dir} — skipping {run_name}",
                UserWarning,
                stacklevel=1,
            )
            continue

        diff_path = run_dir / "difficulty_stats.json"
        hard_path = run_dir / "hard_patch_metrics.json"

        diff = load_json(diff_path)
        hard = load_json(hard_path)

        label = diff_path.as_posix()
        hard_label = hard_path.as_posix()

        row: dict = {
            "run":          run_name,
            "pearson_r":    safe_get(diff, "val_sci_corr_pearson",  label),
            "spearman_rho": safe_get(diff, "val_sci_corr_spearman", label),
            "A_tau":        safe_get(diff, "pct_pred_above_tau",    label),
            "A_hard_tau":   safe_get(diff, "pct_pred_above_tau_on_true_hard", label),
            "n_hard":       safe_get(diff, "n_hard", label),
            "tau":          safe_get(diff, "tau",    label),
            # hard_dice_mean at threshold 0.5 comes from hard_patch_metrics.json;
            # the key is 'hard_dice_mean' (the script applies a fixed 0.5 threshold
            # elsewhere — this value is the mean Dice over is_hard patches)
            "hard_dice_mean_at_0_5": safe_get(hard, "hard_dice_mean", hard_label),
            "activation_note": meta["note"],
        }

        # Warn explicitly about J1 counterfactual status
        if run_name == "J1" and meta["note"]:
            warnings.warn(
                f"[J1 NOTE] {meta['note']}",
                UserWarning,
                stacklevel=1,
            )

        records.append(row)

    if not records:
        print("[ERROR] No run data could be loaded.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(records)

    csv_path = OUT_DIR / "activation_behavior_j1_j2_j3.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved → {csv_path}\n")
    print(df.to_string(index=False))

    # Summarise warnings in stdout for easy scanning
    print("\n--- Notes ---")
    for _, row in df.iterrows():
        if row["activation_note"]:
            print(f"  {row['run']}: {row['activation_note']}")


if __name__ == "__main__":
    main()
