#!/usr/bin/env python3
"""Run all Line A local-search configs sequentially and collect results."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

CONFIGS = [
    "configs/task3_A1_tau069_a065.yaml",
    "configs/task3_A2_tau069_a080.yaml",
    "configs/task3_A3_tau071_a050.yaml",
    "configs/task3_A4_tau071_a065.yaml",
    "configs/task3_A5_tau073_a065.yaml",
    "configs/task3_A6_tau073_a080.yaml",
]

ROOT = Path(__file__).resolve().parents[1]


def run_one(cfg_path: str) -> dict:
    name = Path(cfg_path).stem
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


def load_metrics(out_dir: Path) -> dict:
    row = {"output_dir": str(out_dir)}
    vm = out_dir / "val_metrics.json"
    if vm.is_file():
        d = json.loads(vm.read_text(encoding="utf-8"))
        row["val_dice_mean_at_0_5"] = d.get("val_dice_mean")
        row["hard_dice_mean_at_0_5"] = d.get("hard_dice_mean")
        row["hard_recall_mean_at_0_5"] = d.get("hard_recall_mean")

    tm = out_dir / "threshold_metrics.json"
    if tm.is_file():
        d = json.loads(tm.read_text(encoding="utf-8"))
        best = d.get("best_hard_dice", {})
        row["best_hard_dice_threshold"] = best.get("threshold")
        row["primary_hard_dice_mean"] = best.get("hard_dice_mean")
        row["hard_recall_at_best_hard_dice_threshold"] = best.get("hard_recall_mean")
        best_v = d.get("best_val_dice", {})
        row["best_val_dice_threshold"] = best_v.get("threshold")
        row["best_val_dice_mean"] = best_v.get("val_dice_mean")
    return row


def main():
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    results = []
    for cfg in CONFIGS:
        info = run_one(cfg)
        results.append(info)
        if info["exit_code"] != 0:
            print(f"  WARNING: {info['name']} failed!")

    print(f"\n\n{'='*80}")
    print("  LINE A RESULTS SUMMARY")
    print(f"{'='*80}")

    summary_rows = []
    for info in results:
        if info["exit_code"] != 0:
            print(f"  SKIPPED (failed): {info['name']}")
            continue
        import yaml
        cfg = yaml.safe_load(Path(ROOT / info["config"]).read_text(encoding="utf-8"))
        out_dir = Path(cfg["output_dir"])
        if not out_dir.is_absolute():
            out_dir = ROOT / out_dir
        dw = cfg.get("difficulty_weighting", {})
        metrics = load_metrics(out_dir)
        metrics["run_id"] = info["name"]
        metrics["tau"] = dw.get("tau")
        metrics["alpha"] = dw.get("alpha")
        metrics["max_weight"] = dw.get("max_weight")
        summary_rows.append(metrics)

    print(f"\n{'run_id':<30} {'tau':>6} {'alpha':>6} {'primary_hd':>12} {'hd@0.5':>10} {'val@0.5':>10} {'best_th':>8}")
    print("-" * 90)
    for r in sorted(summary_rows, key=lambda x: x.get("primary_hard_dice_mean", 0), reverse=True):
        print(
            f"{r['run_id']:<30} "
            f"{r.get('tau',''):>6} "
            f"{r.get('alpha',''):>6} "
            f"{r.get('primary_hard_dice_mean', 'n/a'):>12.6f} "
            f"{r.get('hard_dice_mean_at_0_5', 'n/a'):>10.6f} "
            f"{r.get('val_dice_mean_at_0_5', 'n/a'):>10.6f} "
            f"{r.get('best_hard_dice_threshold', 'n/a'):>8}"
        )

    ref_p5 = 0.7374339064055633
    best = max(summary_rows, key=lambda x: x.get("primary_hard_dice_mean", 0))
    print(f"\nBest Line A: {best['run_id']}  primary={best.get('primary_hard_dice_mean', 0):.6f}")
    print(f"P5 reference:                   primary={ref_p5:.6f}")
    delta = best.get("primary_hard_dice_mean", 0) - ref_p5
    print(f"Delta vs P5: {delta:+.6f}")
    if best.get("primary_hard_dice_mean", 0) >= 0.7380:
        print(">>> STOP CONDITION MET: primary >= 0.7380 <<<")

    out_json = ROOT / "outputs" / "line_a_summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
