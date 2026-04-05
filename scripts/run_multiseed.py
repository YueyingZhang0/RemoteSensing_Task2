#!/usr/bin/env python3
"""Phase 1: Run P0/P5/J3 with seeds 40 and 41 for multi-seed confirmation."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ROOT = Path(__file__).resolve().parents[1]

RUNS = [
    # (name, config, output_dir, seed)
    ("P0_baseline_seed40", "configs/task3_v2_1_baseline.yaml", "outputs/task3_P0_baseline_seed40", 40),
    ("P0_baseline_seed41", "configs/task3_v2_1_baseline.yaml", "outputs/task3_P0_baseline_seed41", 41),
    ("P5_oracle_matched_trigger_seed40", "configs/task3_v3_1_oracle_matched_trigger.yaml", "outputs/task3_P5_oracle_matched_trigger_seed40", 40),
    ("P5_oracle_matched_trigger_seed41", "configs/task3_v3_1_oracle_matched_trigger.yaml", "outputs/task3_P5_oracle_matched_trigger_seed41", 41),
    ("J3_joint_detach_false_seed40", "configs/task3_J3_joint_detach_false.yaml", "outputs/task3_J3_joint_detach_false_seed40", 40),
    ("J3_joint_detach_false_seed41", "configs/task3_J3_joint_detach_false.yaml", "outputs/task3_J3_joint_detach_false_seed41", 41),
]


def run_one(name: str, cfg: str, out_dir: str, seed: int) -> dict:
    print(f"\n{'='*60}")
    print(f"  {name}  (seed={seed})")
    print(f"{'='*60}")

    cmd = [
        sys.executable, "-m", "src.main.train_task3",
        "--config", cfg,
        "--model", "baseline",
        "--random_state", str(seed),
        "--output_dir", out_dir,
        "--threshold_sweep",
    ]

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=False)
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else "FAILED"
    print(f"  {name}: {status} in {elapsed:.1f}s")
    return {"name": name, "seed": seed, "exit_code": result.returncode, "elapsed_s": round(elapsed, 1)}


def main():
    results = []
    for name, cfg, out_dir, seed in RUNS:
        info = run_one(name, cfg, out_dir, seed)
        results.append(info)
        if info["exit_code"] != 0:
            print(f"  WARNING: {name} failed with exit code {info['exit_code']}")

    print(f"\n{'='*60}")
    print("  MULTI-SEED RUNS COMPLETE")
    print(f"{'='*60}")
    for r in results:
        status = "OK" if r["exit_code"] == 0 else "FAILED"
        print(f"  {r['name']}: {status} ({r['elapsed_s']}s)")

    failed = [r for r in results if r["exit_code"] != 0]
    if failed:
        print(f"\n  {len(failed)} run(s) FAILED!")
    else:
        print(f"\n  All {len(results)} runs completed successfully.")


if __name__ == "__main__":
    main()
