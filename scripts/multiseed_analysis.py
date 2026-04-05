#!/usr/bin/env python3
"""Phase 2+3: Multi-seed analysis with per-seed table, aggregate, delta, bootstrap CI, fixed-threshold."""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.summarize_task3_v2_1_minimal_matrix import load_row

METHODS = {
    "P0_baseline": [
        ("seed42", "outputs/task3_v2_1_baseline_same_split"),
        ("seed40", "outputs/task3_P0_baseline_seed40"),
        ("seed41", "outputs/task3_P0_baseline_seed41"),
    ],
    "P5_oracle_matched_trigger": [
        ("seed42", "outputs/task3_v3_1_oracle_matched_trigger_same_split"),
        ("seed40", "outputs/task3_P5_oracle_matched_trigger_seed40"),
        ("seed41", "outputs/task3_P5_oracle_matched_trigger_seed41"),
    ],
    "J3_joint_detach_false": [
        ("seed42", "outputs/task3_J3_joint_detach_false"),
        ("seed40", "outputs/task3_J3_joint_detach_false_seed40"),
        ("seed41", "outputs/task3_J3_joint_detach_false_seed41"),
    ],
}

FIELDS = [
    "primary_hard_dice_mean",
    "hard_dice_mean_at_0_5",
    "val_dice_mean_at_0_5",
    "best_hard_dice_threshold",
    "hard_recall_at_best_hard_dice_threshold",
]

PREDICTOR_FIELDS = [
    "val_sci_corr_pearson",
    "val_sci_corr_spearman",
    "pct_pred_above_tau",
    "pct_pred_above_tau_on_true_hard",
]


def load_difficulty_stats(out_dir: Path) -> dict:
    ds = out_dir / "difficulty_stats.json"
    if ds.is_file():
        return json.loads(ds.read_text(encoding="utf-8"))
    return {}


def fmt(x: Any) -> str:
    if x is None:
        return "N/A"
    try:
        return f"{float(x):.6f}"
    except (TypeError, ValueError):
        return str(x)


def fmt4(x: Any) -> str:
    if x is None:
        return "N/A"
    try:
        return f"{float(x):.4f}"
    except (TypeError, ValueError):
        return str(x)


def fmt_pct(x: Any) -> str:
    if x is None:
        return "N/A"
    try:
        return f"{float(x)*100:.1f}%"
    except (TypeError, ValueError):
        return str(x)


def load_threshold_data(out_dir: Path) -> dict:
    tm = out_dir / "threshold_metrics.json"
    if tm.is_file():
        return json.loads(tm.read_text(encoding="utf-8"))
    return {}


def bootstrap_paired_ci(
    dice_a: np.ndarray,
    dice_b: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 999,
) -> dict:
    rng = np.random.RandomState(seed)
    diff = dice_a - dice_b
    n = len(diff)
    boot_means = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot_means[i] = diff[idx].mean()
    alpha = 1.0 - ci
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return {
        "mean_diff": float(diff.mean()),
        "ci_lower": lo,
        "ci_upper": hi,
        "ci_level": ci,
        "n_boot": n_boot,
        "n_samples": n,
        "significant": not (lo <= 0 <= hi),
    }


def collect_per_image_dice(model_dir: Path, threshold: float = 0.5) -> Optional[np.ndarray]:
    """Re-compute per-image Dice from saved model checkpoint on val set."""
    import torch
    from torch.utils.data import DataLoader
    from src.datasets.seg_sce_patch_dataset import SegSCEPatchDataset
    from src.main.train_task3 import _apply_hard_labels
    from src.models.task3_unet import UNetBaseline, UNetBaselineJointDifficulty
    from src.training.splits import load_or_create_group_split
    from src.training.task3_engine import _per_sample_dice_recall, collate_seg_sce
    from src.utils.io import load_yaml

    cfg_path = model_dir / "config_used.yaml"
    ckpt_path = model_dir / "best_model.pt"
    if not cfg_path.is_file() or not ckpt_path.is_file():
        return None

    cfg = load_yaml(cfg_path)
    meta = Path(cfg["metadata_csv"])
    img_dir = Path(cfg["patch_image_dir"])
    msk_dir = Path(cfg["patch_mask_dir"])
    target_col = cfg.get("target_col", "sci_res2_norm")
    group_col = cfg.get("group_col", "sample_id")
    split_json = cfg.get("split_json")
    isz = int(cfg.get("image_size", 128))
    bs = int(cfg.get("batch_size", 16))

    import pandas as pd
    df = pd.read_csv(meta)
    df_tr, df_va = load_or_create_group_split(
        df, group_col, float(cfg.get("val_ratio", 0.2)),
        int(cfg.get("random_state", 42)), split_json, ROOT,
    )
    hard_ratio = float(cfg.get("hard_ratio", 0.2))
    df_tr, df_va = _apply_hard_labels(df_tr, df_va, target_col, hard_ratio)

    val_ds = SegSCEPatchDataset(df_va, img_dir, msk_dir, target_col, isz, augment=False)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0, collate_fn=collate_seg_sce)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dw = cfg.get("difficulty_weighting", {})
    mode = str(dw.get("mode", "")).lower() if dw.get("enabled") else ""
    if mode == "joint":
        model = UNetBaselineJointDifficulty(in_channels=3, base=32).to(device)
    else:
        model = UNetBaseline(in_channels=3, base=32).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_dice = []
    all_hard = []
    with torch.no_grad():
        for batch in val_loader:
            img = batch["image"].to(device)
            m = batch["mask"].to(device)
            hard = batch["is_hard"]
            out = model(img)
            logits = out[0] if isinstance(out, tuple) else out
            d, r = _per_sample_dice_recall(logits, m, thresh=threshold)
            all_dice.extend(d.tolist() if hasattr(d, 'tolist') else list(d))
            all_hard.extend(bool(h) for h in hard)

    return np.asarray(all_dice, dtype=np.float64), np.asarray(all_hard, dtype=bool)


def main():
    print("=" * 100)
    print("  PHASE 2: MULTI-SEED ANALYSIS")
    print("=" * 100)

    all_rows: List[Dict[str, Any]] = []
    method_data: Dict[str, List[Dict[str, Any]]] = {}

    for method, seeds in METHODS.items():
        method_data[method] = []
        for seed_label, out_path in seeds:
            out_dir = ROOT / out_path
            row = load_row(f"{method}_{seed_label}", out_dir)
            ds = load_difficulty_stats(out_dir)
            row["method"] = method
            row["seed"] = seed_label
            row["val_sci_corr_pearson"] = ds.get("val_sci_corr_pearson") or ds.get("pearson")
            row["val_sci_corr_spearman"] = ds.get("val_sci_corr_spearman") or ds.get("spearman")
            row["pct_pred_above_tau"] = ds.get("pct_pred_above_tau") or ds.get("pct_above_tau")
            row["pct_pred_above_tau_on_true_hard"] = ds.get("pct_pred_above_tau_on_true_hard") or ds.get("pct_above_tau_on_hard")
            all_rows.append(row)
            method_data[method].append(row)

    print(f"\n{'='*100}")
    print("  TABLE 1: PER-SEED COMPARISON")
    print(f"{'='*100}")
    header = f"{'Run':<40} {'primary_hd':>12} {'hd@0.5':>10} {'vd@0.5':>10} {'best_th':>8} {'Pearson':>8} {'%>tau_h':>8}"
    print(header)
    print("-" * 100)
    for r in all_rows:
        line = (
            f"{r['run_id']:<40} "
            f"{fmt(r.get('primary_hard_dice_mean')):>12} "
            f"{fmt(r.get('hard_dice_mean_at_0_5')):>10} "
            f"{fmt(r.get('val_dice_mean_at_0_5')):>10} "
            f"{str(r.get('best_hard_dice_threshold','')):>8} "
            f"{fmt4(r.get('val_sci_corr_pearson')):>8} "
            f"{fmt_pct(r.get('pct_pred_above_tau_on_true_hard')):>8}"
        )
        print(line)

    print(f"\n{'='*100}")
    print("  TABLE 2: AGGREGATE SUMMARY (mean +/- std over 3 seeds)")
    print(f"{'='*100}")
    agg_rows = []
    for method in METHODS:
        rows = method_data[method]
        agg: Dict[str, Any] = {"method": method}
        for f in FIELDS:
            vals = [float(r[f]) for r in rows if r.get(f) is not None and not np.isnan(float(r.get(f, float('nan'))))]
            if vals:
                agg[f"{f}_mean"] = float(np.mean(vals))
                agg[f"{f}_std"] = float(np.std(vals))
            else:
                agg[f"{f}_mean"] = None
                agg[f"{f}_std"] = None
        for f in PREDICTOR_FIELDS:
            vals = [float(r[f]) for r in rows if r.get(f) is not None]
            if vals:
                agg[f"{f}_mean"] = float(np.mean(vals))
            else:
                agg[f"{f}_mean"] = None
        agg_rows.append(agg)

    header2 = f"{'Method':<30} {'primary_hd':>18} {'hd@0.5':>18} {'vd@0.5':>18}"
    print(header2)
    print("-" * 90)
    for a in agg_rows:
        pm = a.get("primary_hard_dice_mean_mean")
        ps = a.get("primary_hard_dice_mean_std")
        hm = a.get("hard_dice_mean_at_0_5_mean")
        hs = a.get("hard_dice_mean_at_0_5_std")
        vm = a.get("val_dice_mean_at_0_5_mean")
        vs = a.get("val_dice_mean_at_0_5_std")
        print(
            f"{a['method']:<30} "
            f"{fmt(pm)} +/- {fmt(ps):>8}  "
            f"{fmt(hm)} +/- {fmt(hs):>8}  "
            f"{fmt(vm)} +/- {fmt(vs):>8}"
        )

    print(f"\n{'='*100}")
    print("  TABLE 3: DELTA TABLE (per-seed)")
    print(f"{'='*100}")
    print(f"{'Seed':<10} {'J3-P5 primary':>15} {'J3-P0 primary':>15} {'J3-P5 hd@0.5':>15} {'J3-P0 hd@0.5':>15}")
    print("-" * 75)
    seed_labels = ["seed42", "seed40", "seed41"]
    j3_wins_vs_p5 = 0
    j3_wins_vs_p0 = 0
    for sl in seed_labels:
        j3 = next(r for r in all_rows if r["method"] == "J3_joint_detach_false" and r["seed"] == sl)
        p5 = next(r for r in all_rows if r["method"] == "P5_oracle_matched_trigger" and r["seed"] == sl)
        p0 = next(r for r in all_rows if r["method"] == "P0_baseline" and r["seed"] == sl)
        d_j3_p5 = (j3.get("primary_hard_dice_mean") or 0) - (p5.get("primary_hard_dice_mean") or 0)
        d_j3_p0 = (j3.get("primary_hard_dice_mean") or 0) - (p0.get("primary_hard_dice_mean") or 0)
        d_j3_p5_05 = (j3.get("hard_dice_mean_at_0_5") or 0) - (p5.get("hard_dice_mean_at_0_5") or 0)
        d_j3_p0_05 = (j3.get("hard_dice_mean_at_0_5") or 0) - (p0.get("hard_dice_mean_at_0_5") or 0)
        if d_j3_p5 > 0:
            j3_wins_vs_p5 += 1
        if d_j3_p0 > 0:
            j3_wins_vs_p0 += 1
        print(f"{sl:<10} {d_j3_p5:>+15.6f} {d_j3_p0:>+15.6f} {d_j3_p5_05:>+15.6f} {d_j3_p0_05:>+15.6f}")

    print(f"\nJ3 wins vs P5 on primary: {j3_wins_vs_p5}/3 seeds")
    print(f"J3 wins vs P0 on primary: {j3_wins_vs_p0}/3 seeds")

    agg_j3 = next(a for a in agg_rows if a["method"] == "J3_joint_detach_false")
    agg_p5 = next(a for a in agg_rows if a["method"] == "P5_oracle_matched_trigger")
    agg_p0 = next(a for a in agg_rows if a["method"] == "P0_baseline")
    delta_mean_j3_p5 = (agg_j3.get("primary_hard_dice_mean_mean") or 0) - (agg_p5.get("primary_hard_dice_mean_mean") or 0)
    delta_mean_j3_p0 = (agg_j3.get("primary_hard_dice_mean_mean") or 0) - (agg_p0.get("primary_hard_dice_mean_mean") or 0)
    print(f"\nAggregate mean delta J3-P5 primary: {delta_mean_j3_p5:+.6f}")
    print(f"Aggregate mean delta J3-P0 primary: {delta_mean_j3_p0:+.6f}")

    print(f"\n{'='*100}")
    print("  BOOTSTRAP CI: J3 vs P5 on seed=42 (per-image paired)")
    print(f"{'='*100}")
    j3_dir = ROOT / "outputs" / "task3_J3_joint_detach_false"
    p5_dir = ROOT / "outputs" / "task3_v3_1_oracle_matched_trigger_same_split"
    p0_dir = ROOT / "outputs" / "task3_v2_1_baseline_same_split"

    j3_dice_result = collect_per_image_dice(j3_dir, threshold=0.5)
    p5_dice_result = collect_per_image_dice(p5_dir, threshold=0.5)
    p0_dice_result = collect_per_image_dice(p0_dir, threshold=0.5)

    bootstrap_results = {}
    if j3_dice_result is not None and p5_dice_result is not None:
        j3_dice, j3_hard = j3_dice_result
        p5_dice, _ = p5_dice_result
        ci_all = bootstrap_paired_ci(j3_dice, p5_dice)
        ci_hard = bootstrap_paired_ci(j3_dice[j3_hard], p5_dice[j3_hard])
        print(f"  J3 vs P5 (all val patches):  mean_diff={ci_all['mean_diff']:+.6f}  "
              f"95% CI=[{ci_all['ci_lower']:+.6f}, {ci_all['ci_upper']:+.6f}]  "
              f"sig={ci_all['significant']}")
        print(f"  J3 vs P5 (hard patches):     mean_diff={ci_hard['mean_diff']:+.6f}  "
              f"95% CI=[{ci_hard['ci_lower']:+.6f}, {ci_hard['ci_upper']:+.6f}]  "
              f"sig={ci_hard['significant']}")
        bootstrap_results["J3_vs_P5_all"] = ci_all
        bootstrap_results["J3_vs_P5_hard"] = ci_hard
    else:
        print("  Could not load models for bootstrap CI")

    if j3_dice_result is not None and p0_dice_result is not None:
        p0_dice, _ = p0_dice_result
        ci_all_p0 = bootstrap_paired_ci(j3_dice, p0_dice)
        ci_hard_p0 = bootstrap_paired_ci(j3_dice[j3_hard], p0_dice[j3_hard])
        print(f"\n  J3 vs P0 (all val patches):  mean_diff={ci_all_p0['mean_diff']:+.6f}  "
              f"95% CI=[{ci_all_p0['ci_lower']:+.6f}, {ci_all_p0['ci_upper']:+.6f}]  "
              f"sig={ci_all_p0['significant']}")
        print(f"  J3 vs P0 (hard patches):     mean_diff={ci_hard_p0['mean_diff']:+.6f}  "
              f"95% CI=[{ci_hard_p0['ci_lower']:+.6f}, {ci_hard_p0['ci_upper']:+.6f}]  "
              f"sig={ci_hard_p0['significant']}")
        bootstrap_results["J3_vs_P0_all"] = ci_all_p0
        bootstrap_results["J3_vs_P0_hard"] = ci_hard_p0

    print(f"\n{'='*100}")
    print("  PHASE 3: FIXED-THRESHOLD ROBUSTNESS (seed=42)")
    print(f"{'='*100}")
    fixed_thresholds = [0.30, 0.35, 0.40, 0.50]
    methods_for_fixed = [
        ("P0_baseline", ROOT / "outputs" / "task3_v2_1_baseline_same_split"),
        ("P5_oracle_matched_trigger", ROOT / "outputs" / "task3_v3_1_oracle_matched_trigger_same_split"),
        ("J3_joint_detach_false", ROOT / "outputs" / "task3_J3_joint_detach_false"),
    ]

    for t in fixed_thresholds:
        key = f"{t:.2f}"
        print(f"\n  Threshold = {t}")
        print(f"  {'Method':<30} {'hard_dice':>10} {'hard_recall':>12} {'val_dice':>10}")
        print(f"  {'-'*65}")
        for mname, mdir in methods_for_fixed:
            tm = load_threshold_data(mdir)
            per = tm.get("per_threshold", {}).get(key, {})
            print(
                f"  {mname:<30} "
                f"{fmt(per.get('hard_dice_mean')):>10} "
                f"{fmt(per.get('hard_recall_mean')):>12} "
                f"{fmt(per.get('val_dice_mean')):>10}"
            )

    print(f"\n{'='*100}")
    print("  PASS/FAIL ASSESSMENT")
    print(f"{'='*100}")

    j3_mean = agg_j3.get("primary_hard_dice_mean_mean") or 0
    p5_mean = agg_p5.get("primary_hard_dice_mean_mean") or 0
    p0_mean = agg_p0.get("primary_hard_dice_mean_mean") or 0

    j3_seeds_primary = [float(r.get("primary_hard_dice_mean", 0)) for r in method_data["J3_joint_detach_false"]]
    collapse = any(v < p0_mean - 0.005 for v in j3_seeds_primary)

    strong_pass = j3_mean > p5_mean and not collapse
    weak_pass = abs(j3_mean - p5_mean) < 0.002 and j3_mean > p0_mean and not collapse

    if strong_pass:
        verdict = "STRONG PASS: J3 mean primary > P5 mean primary, no seed collapses"
    elif weak_pass:
        verdict = "WEAK PASS: J3 close to P5, consistently above baseline"
    else:
        verdict = "NOT PASSED: J3 does not consistently outperform P5"

    print(f"\n  J3 mean primary: {j3_mean:.6f}")
    print(f"  P5 mean primary: {p5_mean:.6f}")
    print(f"  P0 mean primary: {p0_mean:.6f}")
    print(f"  J3 wins vs P5:   {j3_wins_vs_p5}/3")
    print(f"  J3 wins vs P0:   {j3_wins_vs_p0}/3")
    print(f"  Seed collapse:   {collapse}")
    print(f"\n  VERDICT: {verdict}")

    out_dir = ROOT / "outputs"
    summary = {
        "per_seed": all_rows,
        "aggregate": agg_rows,
        "delta": {
            "j3_vs_p5_mean_primary": delta_mean_j3_p5,
            "j3_vs_p0_mean_primary": delta_mean_j3_p0,
            "j3_wins_vs_p5": j3_wins_vs_p5,
            "j3_wins_vs_p0": j3_wins_vs_p0,
        },
        "bootstrap_ci": bootstrap_results,
        "verdict": verdict,
    }
    out_json = out_dir / "multiseed_analysis.json"
    out_json.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    keys_csv = ["run_id", "method", "seed"] + FIELDS + PREDICTOR_FIELDS
    out_csv = out_dir / "multiseed_per_seed.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys_csv, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)

    print(f"\n  Saved: {out_json}")
    print(f"  Saved: {out_csv}")


if __name__ == "__main__":
    main()
