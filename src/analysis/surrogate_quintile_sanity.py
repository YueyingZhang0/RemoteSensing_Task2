"""
Surrogate quintile sanity check.

Divides the 172 validation patches into five quintiles by sci_res2_norm and
computes P0 Dice statistics per quintile.  Also reports the Spearman
correlation between sci_res2_norm and p0_dice across all patches.

Input : outputs/task3_J3_patch_audit/j3_patch_audit.csv
Output: outputs/revision_analysis/surrogate_quintile_sanity.csv
        outputs/revision_analysis/surrogate_quintile_sanity.png
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
AUDIT_CSV = ROOT / "outputs" / "task3_J3_patch_audit" / "j3_patch_audit.csv"
OUT_DIR   = ROOT / "outputs" / "revision_analysis"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load audit CSV
    # ------------------------------------------------------------------
    if not AUDIT_CSV.exists():
        print(f"[ERROR] Input file not found: {AUDIT_CSV}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(AUDIT_CSV)
    print(f"Loaded {len(df)} patches from {AUDIT_CSV.name}")

    required = {"sci_res2_norm", "p0_dice"}
    missing = required - set(df.columns)
    if missing:
        print(f"[ERROR] Required columns missing from CSV: {missing}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Optional clDice column
    # ------------------------------------------------------------------
    has_cldice = "p0_cldice" in df.columns
    if not has_cldice:
        warnings.warn(
            "Column 'p0_cldice' not found in audit CSV — "
            "clDice quintile statistics will be skipped.",
            UserWarning,
            stacklevel=1,
        )

    # ------------------------------------------------------------------
    # Quintile assignment
    # ------------------------------------------------------------------
    df["quintile"] = pd.qcut(df["sci_res2_norm"], q=5, labels=False, duplicates="drop")
    n_quintiles = df["quintile"].nunique()
    if n_quintiles < 5:
        warnings.warn(
            f"Only {n_quintiles} distinct quintile bins created "
            "(possible ties in sci_res2_norm).",
            UserWarning,
        )

    # ------------------------------------------------------------------
    # Per-quintile statistics
    # ------------------------------------------------------------------
    records = []
    for q_idx in sorted(df["quintile"].unique()):
        sub = df[df["quintile"] == q_idx]
        row: dict = {
            "quintile": int(q_idx) + 1,
            "n": len(sub),
            "sci_res2_norm_min":  sub["sci_res2_norm"].min(),
            "sci_res2_norm_max":  sub["sci_res2_norm"].max(),
            "sci_res2_norm_mean": sub["sci_res2_norm"].mean(),
            "p0_dice_mean":       sub["p0_dice"].mean(),
            "p0_dice_std":        sub["p0_dice"].std(ddof=1),
        }
        if has_cldice:
            row["p0_cldice_mean"] = sub["p0_cldice"].mean()
            row["p0_cldice_std"]  = sub["p0_cldice"].std(ddof=1)
        records.append(row)

    result_df = pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Spearman correlation: sci_res2_norm ↔ p0_dice
    # ------------------------------------------------------------------
    rho, pval = spearmanr(df["sci_res2_norm"], df["p0_dice"])
    print(
        f"\nSpearman ρ(sci_res2_norm, p0_dice) = {rho:.4f}  "
        f"(p = {pval:.4g}, n = {len(df)})"
    )

    spearman_row = pd.DataFrame([{
        "quintile": "ALL",
        "n": len(df),
        "sci_res2_norm_min":  df["sci_res2_norm"].min(),
        "sci_res2_norm_max":  df["sci_res2_norm"].max(),
        "sci_res2_norm_mean": df["sci_res2_norm"].mean(),
        "p0_dice_mean": df["p0_dice"].mean(),
        "p0_dice_std":  df["p0_dice"].std(ddof=1),
        "spearman_rho": rho,
        "spearman_pval": pval,
    }])
    result_df = pd.concat([result_df, spearman_row], ignore_index=True)

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    csv_path = OUT_DIR / "surrogate_quintile_sanity.csv"
    result_df.to_csv(csv_path, index=False)
    print(f"\nSaved table → {csv_path}")
    print(result_df.to_string(index=False))

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    quintile_rows = result_df[result_df["quintile"] != "ALL"].copy()
    quintile_rows["quintile"] = quintile_rows["quintile"].astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: bar chart — mean P0 Dice per quintile
    ax = axes[0]
    xs = quintile_rows["quintile"].tolist()
    ys = quintile_rows["p0_dice_mean"].astype(float).tolist()
    errs = quintile_rows["p0_dice_std"].astype(float).tolist()
    ns  = quintile_rows["n"].astype(int).tolist()
    bars = ax.bar(xs, ys, yerr=errs, capsize=4, color="steelblue", alpha=0.8,
                  error_kw={"elinewidth": 1.2})
    ax.set_xticks(xs)
    ax.set_xticklabels([f"Q{x}\n(n={n})" for x, n in zip(xs, ns)])
    ax.set_xlabel("Quintile (ascending sci_res2_norm)")
    ax.set_ylabel("Mean P0 Dice")
    ax.set_title("P0 Dice by sci_res2_norm quintile")
    # Annotate mean values above bars
    for bar, y in zip(bars, ys):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(errs) * 0.05,
            f"{y:.3f}",
            ha="center", va="bottom", fontsize=8,
        )
    # Add Spearman annotation
    ax.annotate(
        f"Spearman ρ = {rho:.3f}\np = {pval:.3g}  (n={len(df)})",
        xy=(0.98, 0.02), xycoords="axes fraction",
        ha="right", va="bottom", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.8),
    )

    # Right: scatter plot — sci_res2_norm vs p0_dice
    ax2 = axes[1]
    ax2.scatter(df["sci_res2_norm"], df["p0_dice"],
                alpha=0.4, s=18, color="steelblue", edgecolors="none")
    # Trend line
    z = np.polyfit(df["sci_res2_norm"], df["p0_dice"], 1)
    xline = np.linspace(df["sci_res2_norm"].min(), df["sci_res2_norm"].max(), 200)
    ax2.plot(xline, np.polyval(z, xline), color="firebrick", linewidth=1.5,
             label="Linear fit")
    ax2.set_xlabel("sci_res2_norm")
    ax2.set_ylabel("P0 Dice")
    ax2.set_title("P0 Dice vs surrogate score")
    ax2.annotate(
        f"Spearman ρ = {rho:.3f}\np = {pval:.3g}  (n={len(df)})",
        xy=(0.02, 0.02), xycoords="axes fraction",
        ha="left", va="bottom", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.8),
    )

    fig.suptitle(
        "Surrogate score quintile sanity: sci_res2_norm vs P0 Dice\n"
        "(J3 audit CSV, 172 validation patches)",
        fontsize=11,
    )
    fig.tight_layout()
    png_path = OUT_DIR / "surrogate_quintile_sanity.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot  → {png_path}")


if __name__ == "__main__":
    main()
