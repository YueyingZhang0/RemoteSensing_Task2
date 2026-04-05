#!/usr/bin/env python3
"""Run Line B joint experiments sequentially and collect results."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CONFIGS = [
    ("J1_joint_calib_only", "configs/task3_J1_joint_calib_only.yaml"),
    ("J2_joint_detach_true", "configs/task3_J2_joint_detach_true.yaml"),
    ("J3_joint_detach_false", "configs/task3_J3_joint_detach_false.yaml"),
]


def run_one(name: str, cfg_path: str) -> dict:
    print(f"\n{'='*60}")
    print(f"  Running: {name}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "-m", "src.main.train_task3", "--config", cfg_path, "--model", "baseline"],
        cwd=str(ROOT),
        capture_output=False,
    )
    elapsed = time.time() - t0
    print(f"  {name} finished in {elapsed:.1f}s  (exit={result.returncode})")
    return {"config": cfg_path, "name": name, "exit_code": result.returncode, "elapsed_s": round(elapsed, 1)}


def load_joint_metrics(out_dir: Path) -> dict:
    """Load metrics + difficulty stats for a joint run."""
    from src.analysis.summarize_task3_v2_1_minimal_matrix import load_row
    row = load_row("", out_dir)

    vm = out_dir / "val_metrics.json"
    if vm.is_file():
        v = json.loads(vm.read_text(encoding="utf-8"))
        ds = v.get("difficulty_stats", {})
        row["val_sci_corr_pearson"] = ds.get("pearson")
        row["val_sci_corr_spearman"] = ds.get("spearman")
        row["pred_sci_mean"] = ds.get("pred_mean")
        row["pred_sci_std"] = ds.get("pred_std")
        row["pct_pred_above_tau"] = ds.get("pct_above_tau")
        row["pct_pred_above_tau_on_true_hard"] = ds.get("pct_above_tau_on_hard")

    return row


def main():
    results = []
    for name, cfg in CONFIGS:
        info = run_one(name, cfg)
        results.append(info)
        if info["exit_code"] != 0:
            print(f"  ERROR: {name} failed!")
            if name == "J2_joint_detach_true":
                print("  J2 failed => skipping J3")
                break

        if name == "J2_joint_detach_true" and info["exit_code"] == 0:
            out_j2 = ROOT / "outputs" / "task3_J2_joint_detach_true"
            hist = out_j2 / "history.json"
            if hist.is_file():
                h = json.loads(hist.read_text(encoding="utf-8"))
                last_losses = [e.get("train_seg_loss", 999) for e in h[-5:]] if isinstance(h, list) else []
                if any(l > 10 for l in last_losses):
                    print("  WARNING: J2 appears unstable (high loss at end). Skipping J3.")
                    break

    print(f"\n\n{'='*80}")
    print("  LINE B RESULTS SUMMARY")
    print(f"{'='*80}")

    fmt = lambda x: f"{x:.6f}" if x is not None else "n/a"

    for info in results:
        if info["exit_code"] != 0:
            print(f"\n  {info['name']}: FAILED (exit={info['exit_code']})")
            continue
        import yaml
        cfg = yaml.safe_load(Path(ROOT / info["config"]).read_text(encoding="utf-8"))
        out_dir = Path(cfg["output_dir"])
        if not out_dir.is_absolute():
            out_dir = ROOT / out_dir
        m = load_joint_metrics(out_dir)
        m["run_id"] = info["name"]

        print(f"\n  {info['name']}:")
        print(f"    primary_hard_dice_mean:           {fmt(m.get('primary_hard_dice_mean'))}")
        print(f"    hard_dice_mean @ 0.5:             {fmt(m.get('hard_dice_mean_at_0_5'))}")
        print(f"    val_dice_mean @ 0.5:              {fmt(m.get('val_dice_mean_at_0_5'))}")
        print(f"    best_hard_dice_threshold:          {m.get('best_hard_dice_threshold')}")
        print(f"    val_sci_corr_pearson:              {fmt(m.get('val_sci_corr_pearson'))}")
        print(f"    val_sci_corr_spearman:             {fmt(m.get('val_sci_corr_spearman'))}")
        print(f"    pct_pred_above_tau:                {fmt(m.get('pct_pred_above_tau'))}")
        print(f"    pct_pred_above_tau_on_true_hard:   {fmt(m.get('pct_pred_above_tau_on_true_hard'))}")

    ref_p4 = 0.7361483200233386
    ref_p5 = 0.7374339064055633
    print(f"\n  Reference: P4 frozen_probe primary={ref_p4:.6f}")
    print(f"  Reference: P5 oracle_matched_trigger primary={ref_p5:.6f}")


if __name__ == "__main__":
    main()
