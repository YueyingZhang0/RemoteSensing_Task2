#!/usr/bin/env python3
"""
Generate per-patch Dice for J1 and J2 on the canonical 172 validation patches,
merge with the existing j3_patch_audit.csv (which already contains j3_dice, p0_dice,
p5_dice), and run paired bootstrap for all four comparisons on the hard subset.

Pre-flight config compatibility is verified before any model is loaded.

Outputs
-------
outputs/revision_analysis/canonical_patch_audit_all_methods.csv
outputs/revision_analysis/bootstrap_j3_vs_j1_hard.json
outputs/revision_analysis/bootstrap_j3_vs_j2_hard.json
outputs/revision_analysis/bootstrap_j3_vs_p0_hard.json   (re-run for consistency)
outputs/revision_analysis/bootstrap_j3_vs_p5_hard.json   (re-run for consistency)

Usage
-----
python src/analysis/generate_j1_j2_patch_dice.py \\
    [--j1_ckpt outputs/task3_J1_joint_calib_only/best_model.pt] \\
    [--j2_ckpt outputs/task3_J2_joint_detach_true/best_model.pt] \\
    [--threshold 0.5] [--n_boot 10000] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.datasets.seg_sce_patch_dataset import SegSCEPatchDataset
from src.main.train_task3 import _apply_hard_labels
from src.models.task3_unet import UNetBaselineJointDifficulty
from src.training.splits import load_or_create_group_split
from src.training.task3_engine import _per_sample_dice_recall, collate_seg_sce
from src.utils.io import load_yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
J3_CONFIG   = ROOT / "configs" / "task3_J3_joint_detach_false.yaml"
AUDIT_CSV   = ROOT / "outputs" / "task3_J3_patch_audit" / "j3_patch_audit.csv"
OUT_DIR     = ROOT / "outputs" / "revision_analysis"

EXPECTED_N_VAL  = 172
EXPECTED_N_HARD = 35

# Fields that MUST be identical across all three run configs
MUST_MATCH = [
    "split_json", "metadata_csv", "patch_image_dir", "patch_mask_dir",
    "target_col", "group_col", "val_ratio", "random_state",
    "hard_ratio", "image_size", "model",
]


# ---------------------------------------------------------------------------
# Config compatibility check
# ---------------------------------------------------------------------------

def _check_configs() -> dict:
    """Load and compare configs for J1/J2/J3. Stop if any MUST_MATCH field differs."""
    cfg_paths = {
        "J1": ROOT / "outputs" / "task3_J1_joint_calib_only"  / "config_used.yaml",
        "J2": ROOT / "outputs" / "task3_J2_joint_detach_true"  / "config_used.yaml",
        "J3": ROOT / "outputs" / "task3_J3_joint_detach_false" / "config_used.yaml",
    }
    cfgs: Dict[str, dict] = {}
    for name, path in cfg_paths.items():
        if not path.exists():
            print(f"[ERROR] Config not found: {path}", file=sys.stderr)
            sys.exit(1)
        cfgs[name] = load_yaml(path)

    print("\n=== Config compatibility check ===")
    mismatches: List[str] = []
    ref = cfgs["J3"]
    for field in MUST_MATCH:
        vals = {name: cfgs[name].get(field) for name in cfgs}
        ref_val = ref.get(field)
        ok = all(v == ref_val for v in vals.values())
        status = "OK" if ok else "MISMATCH"
        print(f"  {field:<28} {status}  {vals}")
        if not ok:
            mismatches.append(field)

    ckpt_sizes = {}
    for name in ("J1", "J2", "J3"):
        run_dir = cfg_paths[name].parent
        pt = run_dir / "best_model.pt"
        ckpt_sizes[name] = pt.stat().st_size if pt.exists() else -1
    size_ok = len(set(ckpt_sizes.values())) == 1
    print(f"  {'checkpoint_bytes':<28} {'OK' if size_ok else 'NOTE'}  {ckpt_sizes}")
    if not size_ok:
        print("  NOTE: checkpoint sizes differ — architectures may differ; proceeding anyway.")

    if mismatches:
        print(
            f"\n[ABORT] Config mismatch in: {mismatches}. "
            "Cannot guarantee the same validation set across runs.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("  => All critical fields match. Proceeding.\n")
    return ref   # return J3 config as canonical


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_joint_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    """Load UNetBaselineJointDifficulty from a checkpoint."""
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)
    model = UNetBaselineJointDifficulty(in_channels=3, base=32).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Loaded {ckpt_path.name}  "
          f"(epoch={ckpt.get('epoch', '?')}, "
          f"val_dice={ckpt.get('val_dice', float('nan')):.4f})")
    return model


# ---------------------------------------------------------------------------
# Per-patch inference
# ---------------------------------------------------------------------------

def collect_per_patch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> pd.DataFrame:
    all_dice, all_names = [], []
    with torch.no_grad():
        for batch in loader:
            img  = batch["image"].to(device)
            mask = batch["mask"].to(device)
            out  = model(img)
            logits = out[0] if isinstance(out, tuple) else out
            d, _ = _per_sample_dice_recall(logits, mask, thresh=threshold)
            all_dice.extend(d.tolist() if hasattr(d, "tolist") else list(d))
            all_names.extend(
                batch.get("patch_name", [f"patch_{i}" for i in range(len(d))])
            )
    return pd.DataFrame({"patch_name": all_names, "dice": all_dice})


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def paired_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 10_000,
    rng: np.random.Generator | None = None,
) -> dict:
    """Paired bootstrap: H1: mean(a - b) > 0  (one-sided)."""
    if rng is None:
        rng = np.random.default_rng()
    diff = a - b
    obs  = float(np.mean(diff))
    n    = len(diff)
    boot = np.array([
        rng.choice(diff, size=n, replace=True).mean()
        for _ in range(n_boot)
    ])
    ci_lo = float(np.percentile(boot, 2.5))
    ci_hi = float(np.percentile(boot, 97.5))
    p_one = float(np.mean(boot <= 0.0))
    return {
        "observed_diff":     obs,
        "ci_lower_2p5":      ci_lo,
        "ci_upper_97p5":     ci_hi,
        "p_value_one_sided": p_one,
        "n_patches":         n,
        "n_boot":            n_boot,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--j1_ckpt",    default="outputs/task3_J1_joint_calib_only/best_model.pt")
    ap.add_argument("--j2_ckpt",    default="outputs/task3_J2_joint_detach_true/best_model.pt")
    ap.add_argument("--threshold",  type=float, default=0.5)
    ap.add_argument("--n_boot",     type=int,   default=10_000)
    ap.add_argument("--seed",       type=int,   default=42)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Config compatibility check
    # ------------------------------------------------------------------
    cfg = _check_configs()

    # ------------------------------------------------------------------
    # 2. Load canonical audit CSV (ground-truth patch list)
    # ------------------------------------------------------------------
    if not AUDIT_CSV.exists():
        print(f"[ERROR] Canonical audit CSV not found: {AUDIT_CSV}", file=sys.stderr)
        sys.exit(1)

    ref_df = pd.read_csv(AUDIT_CSV)
    print(f"Canonical audit CSV: {len(ref_df)} patches, "
          f"{ref_df['is_hard'].sum()} hard")

    for col in ("patch_name", "is_hard", "sci_res2_norm",
                "p0_dice", "p5_dice", "j3_dice"):
        if col not in ref_df.columns:
            print(f"[ERROR] Required column '{col}' missing from audit CSV.",
                  file=sys.stderr)
            sys.exit(1)

    if len(ref_df) != EXPECTED_N_VAL:
        print(
            f"[ERROR] Expected {EXPECTED_N_VAL} validation patches, "
            f"got {len(ref_df)}.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Build validation DataLoader using canonical config
    # ------------------------------------------------------------------
    target_col    = cfg["target_col"]
    group_col     = cfg["group_col"]
    hard_ratio    = float(cfg["hard_ratio"])
    val_ratio     = float(cfg["val_ratio"])
    random_state  = int(cfg["random_state"])
    split_json    = cfg.get("split_json")
    isz           = int(cfg["image_size"])
    bs            = int(cfg["batch_size"])

    meta_path = Path(cfg["metadata_csv"])
    if not meta_path.is_absolute():
        meta_path = ROOT / meta_path
    img_dir = Path(cfg["patch_image_dir"])
    if not img_dir.is_absolute():
        img_dir = ROOT / img_dir
    msk_dir = Path(cfg["patch_mask_dir"])
    if not msk_dir.is_absolute():
        msk_dir = ROOT / msk_dir

    meta_full = pd.read_csv(meta_path)
    _, df_val = load_or_create_group_split(
        meta_full, group_col, val_ratio, random_state, split_json, ROOT,
    )
    _, df_val = _apply_hard_labels(
        meta_full[~meta_full.index.isin(df_val.index)].reset_index(drop=True),
        df_val.reset_index(drop=True),
        target_col, hard_ratio,
    )

    val_ds = SegSCEPatchDataset(
        df_val, img_dir, msk_dir, target_col, isz, augment=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=0, collate_fn=collate_seg_sce,
    )
    print(f"Val DataLoader: {len(val_ds)} patches")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ------------------------------------------------------------------
    # 4. Run inference for J1 and J2
    # ------------------------------------------------------------------
    j1_path = ROOT / args.j1_ckpt
    j2_path = ROOT / args.j2_ckpt

    print("Loading J1 model...")
    j1_model = load_joint_model(j1_path, device)
    print("Running J1 inference...")
    j1_df = collect_per_patch(j1_model, val_loader, device, args.threshold)
    del j1_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nLoading J2 model...")
    j2_model = load_joint_model(j2_path, device)
    print("Running J2 inference...")
    j2_df = collect_per_patch(j2_model, val_loader, device, args.threshold)
    del j2_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 5. Merge by patch_name (not by row order)
    # ------------------------------------------------------------------
    j1_df = j1_df.rename(columns={"dice": "j1_dice"})
    j2_df = j2_df.rename(columns={"dice": "j2_dice"})

    # Duplicate detection
    for name, sub in [("J1", j1_df), ("J2", j2_df)]:
        dups = sub["patch_name"].duplicated().sum()
        if dups:
            print(f"[ERROR] {name} inference produced {dups} duplicate patch_names.",
                  file=sys.stderr)
            sys.exit(1)

    merged = (
        ref_df[["patch_name", "is_hard", "sci_res2_norm",
                "p0_dice", "p5_dice", "j3_dice"]]
        .merge(j1_df[["patch_name", "j1_dice"]], on="patch_name", how="left")
        .merge(j2_df[["patch_name", "j2_dice"]], on="patch_name", how="left")
    )

    # ------------------------------------------------------------------
    # 6. Validate merge
    # ------------------------------------------------------------------
    missing_j1 = merged["j1_dice"].isna().sum()
    missing_j2 = merged["j2_dice"].isna().sum()
    if missing_j1 or missing_j2:
        print(
            f"[ERROR] patch_name matching failed: "
            f"{missing_j1} patches have no J1 Dice, "
            f"{missing_j2} patches have no J2 Dice.\n"
            "This means the validation set from the DataLoader does not fully "
            "match the canonical audit CSV. Aborting.",
            file=sys.stderr,
        )
        sys.exit(1)

    n_val  = len(merged)
    n_hard = int(merged["is_hard"].sum())

    if n_val != EXPECTED_N_VAL:
        print(f"[ERROR] Expected {EXPECTED_N_VAL} patches after merge, got {n_val}.",
              file=sys.stderr)
        sys.exit(1)
    if n_hard != EXPECTED_N_HARD:
        print(
            f"[ERROR] Expected {EXPECTED_N_HARD} hard patches after merge, "
            f"got {n_hard}. Aborting.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\nMerge OK: {n_val} patches, {n_hard} hard")

    # Reorder columns
    merged = merged[[
        "patch_name", "is_hard", "sci_res2_norm",
        "p0_dice", "p5_dice", "j1_dice", "j2_dice", "j3_dice",
    ]]

    # ------------------------------------------------------------------
    # 7. Save canonical CSV
    # ------------------------------------------------------------------
    out_csv = OUT_DIR / "canonical_patch_audit_all_methods.csv"
    merged.to_csv(out_csv, index=False)
    print(f"Saved canonical CSV → {out_csv}")

    # Quick sanity: compare J3 dice in merged vs what was in the original CSV
    j3_delta = (merged["j3_dice"] - ref_df.set_index("patch_name")
                .loc[merged["patch_name"], "j3_dice"].values).abs().max()
    print(f"  Max |Δ j3_dice| between original audit and merged CSV: {j3_delta:.2e}")

    # ------------------------------------------------------------------
    # 8. Paired bootstrap on hard subset only
    # ------------------------------------------------------------------
    hard = merged[merged["is_hard"] == True].copy()
    assert len(hard) == EXPECTED_N_HARD, \
        f"Hard subset has {len(hard)} rows, expected {EXPECTED_N_HARD}"

    rng = np.random.default_rng(args.seed)

    comparisons = [
        ("J3", "J1", hard["j3_dice"].values, hard["j1_dice"].values),
        ("J3", "J2", hard["j3_dice"].values, hard["j2_dice"].values),
        ("J3", "P0", hard["j3_dice"].values, hard["p0_dice"].values),
        ("J3", "P5", hard["j3_dice"].values, hard["p5_dice"].values),
    ]

    summary_rows = []
    for a_label, b_label, a, b in comparisons:
        res = paired_bootstrap(a, b, n_boot=args.n_boot, rng=rng)
        res["comparison"]   = f"{a_label}_vs_{b_label}"
        res["subset"]       = "hard"
        res[f"{a_label.lower()}_mean_hard"] = float(np.mean(a))
        res[f"{b_label.lower()}_mean_hard"] = float(np.mean(b))
        if b_label == "J1":
            res["note"] = ("J1 activation is COUNTERFACTUAL (dynamic weighting not "
                           "active during training); Dice is a real model prediction.")

        fname = f"bootstrap_{a_label.lower()}_vs_{b_label.lower()}_hard.json"
        out_path = OUT_DIR / fname
        with out_path.open("w") as fh:
            json.dump(res, fh, indent=2)

        sig = res["p_value_one_sided"] < 0.05
        summary_rows.append({
            "comparison":    f"{a_label} vs {b_label}",
            "n":             res["n_patches"],
            "obs_diff":      res["observed_diff"],
            "ci95_low":      res["ci_lower_2p5"],
            "ci95_high":     res["ci_upper_97p5"],
            "p_one_sided":   res["p_value_one_sided"],
            "significant":   sig,
        })

    # ------------------------------------------------------------------
    # 9. Print summary table
    # ------------------------------------------------------------------
    print(f"\n{'='*74}")
    print(f"Paired bootstrap on hard patches (n={EXPECTED_N_HARD}, "
          f"seed={args.seed}, n_boot={args.n_boot})")
    print(f"{'='*74}")
    hdr = (f"{'comparison':<16} {'n':>4}  {'obs_diff':>9}  "
           f"{'ci95_low':>9}  {'ci95_high':>9}  {'p_one_sided':>11}  {'sig?':>5}")
    print(hdr)
    print("-" * 74)
    for row in summary_rows:
        sig_str = "YES *" if row["significant"] else "no"
        print(
            f"{row['comparison']:<16} {row['n']:>4}  {row['obs_diff']:>+9.5f}  "
            f"{row['ci95_low']:>+9.5f}  {row['ci95_high']:>+9.5f}  "
            f"{row['p_one_sided']:>11.4f}  {sig_str:>5}"
        )
    print(f"{'='*74}")
    print(
        "\nNOTE: J1 activation is COUNTERFACTUAL — dynamic weighting was not\n"
        "  active during J1 training. J1 Dice reflects genuine model predictions\n"
        "  from a model that did not actually use difficulty-based weighting.\n"
        "  J3 vs J1 therefore measures the full joint-training effect.\n"
    )

    j1_bootstrap_path = OUT_DIR / "bootstrap_j3_vs_j1_hard.json"
    print(f"All bootstrap JSONs saved to: {OUT_DIR}")
    print(f"Canonical CSV: {out_csv}")


if __name__ == "__main__":
    main()
