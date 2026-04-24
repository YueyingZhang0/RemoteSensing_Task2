"""
Paired bootstrap: J3 vs J1 / J2 / P0 / P5.

INTEGRITY GUARD
---------------
This script will NOT run J3 vs J1 or J3 vs J2 unless per-patch Dice columns
('j1_dice' and 'j2_dice' respectively) are present in the audit CSV.
These values CANNOT be fabricated from run-level means — doing so would
invalidate the paired test.

Currently available per-patch columns in j3_patch_audit.csv:
    j3_dice, p0_dice, p5_dice   →  J3 vs P0 and J3 vs P5 are SUPPORTED.
    j1_dice, j2_dice            →  NOT YET PRESENT; those comparisons are BLOCKED.

Usage
-----
    python src/analysis/bootstrap_j3_vs_j2_j1.py [--n-boot N] [--seed S]

Outputs (when available comparisons exist):
    outputs/revision_analysis/bootstrap_j3_vs_p0.json
    outputs/revision_analysis/bootstrap_j3_vs_p5.json
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT      = Path(__file__).resolve().parents[2]
AUDIT_CSV = ROOT / "outputs" / "task3_J3_patch_audit" / "j3_patch_audit.csv"
OUT_DIR   = ROOT / "outputs" / "revision_analysis"

# ---------------------------------------------------------------------------
# Column → label mapping
# ---------------------------------------------------------------------------
# Keys are column names in the audit CSV; values are human-readable labels.
COMPARISONS = {
    "p0_dice": "P0",
    "p5_dice": "P5",
    "j2_dice": "J2",
    "j1_dice": "J1",
}

# Columns whose absence BLOCKS execution rather than merely skips the comparison.
# J1 and J2 per-patch Dice are not yet recorded in the audit CSV.
BLOCKED_MISSING: set[str] = {"j1_dice", "j2_dice"}


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def paired_bootstrap_mean_diff(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 10_000,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    One-sided paired bootstrap for H0: mean(a - b) <= 0  (i.e., a > b?).

    Returns
    -------
    dict with keys: observed_diff, ci_lower, ci_upper, p_value_one_sided,
                    n_patches, n_boot
    """
    if rng is None:
        rng = np.random.default_rng()

    diff = a - b
    obs  = float(np.mean(diff))
    n    = len(diff)

    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = diff[idx].mean()

    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))

    # p-value: fraction of bootstrap samples with mean <= 0 (one-sided, H1: a > b)
    p_val = float(np.mean(boot_means <= 0.0))

    return {
        "observed_diff":       obs,
        "ci_lower_2p5":        ci_lower,
        "ci_upper_97p5":       ci_upper,
        "p_value_one_sided":   p_val,
        "n_patches":           n,
        "n_boot":              n_boot,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=10_000,
                        help="Number of bootstrap resamples (default: 10000)")
    parser.add_argument("--seed",   type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--hard-only", action="store_true", default=False,
                        help="Restrict bootstrap to is_hard==True patches (n=35). "
                             "This is the methodologically correct subset: J3 training "
                             "explicitly targets these patches via dynamic weighting.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load audit CSV
    # ------------------------------------------------------------------
    if not AUDIT_CSV.exists():
        print(f"[ERROR] Audit CSV not found: {AUDIT_CSV}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(AUDIT_CSV)
    n_total = len(df)
    present_cols = set(df.columns)
    print(f"Loaded {n_total} patches from {AUDIT_CSV.name}")

    # ------------------------------------------------------------------
    # Optional hard-patch filter
    # ------------------------------------------------------------------
    if args.hard_only:
        if "is_hard" not in present_cols:
            print("[ERROR] --hard-only requested but 'is_hard' column is absent.",
                  file=sys.stderr)
            sys.exit(1)
        df = df[df["is_hard"] == True].copy()
        print(
            f"Filtered to is_hard==True: {len(df)}/{n_total} patches "
            f"(J3 dynamic weighting targets this subset)"
        )
        subset_tag = "hard"
    else:
        print(
            f"Using all {n_total} patches (no is_hard filter). "
            "Pass --hard-only to restrict to the 35 hard patches."
        )
        subset_tag = "all"

    present_cols = set(df.columns)   # refresh after potential filter
    print(f"Available Dice columns: {sorted(present_cols & set(COMPARISONS.keys()))}")

    # ------------------------------------------------------------------
    # HARD BLOCK: refuse if blocked columns are absent
    # ------------------------------------------------------------------
    blocked_absent = BLOCKED_MISSING - present_cols
    if blocked_absent:
        for col in sorted(blocked_absent):
            label = COMPARISONS[col]
            warnings.warn(
                f"[BLOCKED] Column '{col}' ({label} per-patch Dice) is absent from "
                f"the audit CSV.  J3 vs {label} bootstrap CANNOT run — per-patch "
                f"values must not be fabricated from run-level means.",
                UserWarning,
                stacklevel=1,
            )

    # ------------------------------------------------------------------
    # Determine which comparisons CAN run
    # ------------------------------------------------------------------
    runnable = {
        col: label
        for col, label in COMPARISONS.items()
        if col not in BLOCKED_MISSING and col in present_cols
    }

    if not runnable:
        print(
            "\n[ABORT] No supported comparisons are available with current data.\n"
            "  Blocked (columns absent, must not fabricate): "
            + ", ".join(sorted(blocked_absent))
            + "\n  Skipped (not in CSV): "
            + ", ".join(
                sorted(set(COMPARISONS) - BLOCKED_MISSING - present_cols)
            ),
            file=sys.stderr,
        )
        sys.exit(1)

    if "j3_dice" not in present_cols:
        print("[ERROR] Column 'j3_dice' not found — cannot run any comparison.",
              file=sys.stderr)
        sys.exit(1)

    j3 = df["j3_dice"].values
    rng = np.random.default_rng(args.seed)

    # ------------------------------------------------------------------
    # Run bootstrap for each supported comparison
    # ------------------------------------------------------------------
    results = {}
    for col, label in runnable.items():
        other = df[col].values
        res = paired_bootstrap_mean_diff(j3, other, n_boot=args.n_boot, rng=rng)
        res["comparison"]  = f"J3_vs_{label}"
        res["subset"]      = subset_tag          # "hard" or "all"
        res["n_total_csv"] = n_total
        res["j3_mean"]     = float(np.mean(j3))
        res[f"{label.lower()}_mean"] = float(np.mean(other))
        results[f"J3_vs_{label}"] = res

        print(
            f"\nJ3 vs {label} [{subset_tag}]:  obs_diff={res['observed_diff']:+.5f}  "
            f"95% CI [{res['ci_lower_2p5']:+.5f}, {res['ci_upper_97p5']:+.5f}]  "
            f"p(one-sided)={res['p_value_one_sided']:.4f}  n={res['n_patches']}"
        )

        out_path = OUT_DIR / f"bootstrap_j3_vs_{label.lower()}_{subset_tag}.json"
        with out_path.open("w") as fh:
            json.dump(res, fh, indent=2)
        print(f"Saved → {out_path}")

    # ------------------------------------------------------------------
    # Summarise what was skipped / blocked
    # ------------------------------------------------------------------
    if blocked_absent:
        print(
            "\n[SUMMARY] Blocked comparisons (per-patch data unavailable):\n"
            + "\n".join(
                f"  J3 vs {COMPARISONS[c]}: column '{c}' missing — "
                "DO NOT FABRICATE from run-level means"
                for c in sorted(blocked_absent)
            )
        )


if __name__ == "__main__":
    main()
