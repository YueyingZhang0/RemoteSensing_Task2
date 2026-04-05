#!/usr/bin/env python3
"""Run P0/P5/J3 on splitB with seed=40 to resolve ambiguous ranking."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
ROOT = Path(__file__).resolve().parents[1]

RUNS = [
    ("P0_baseline_splitB_seed40", "configs/_tmp_splitB_seed40_P0.yaml"),
    ("P5_oracle_matched_trigger_splitB_seed40", "configs/_tmp_splitB_seed40_P5.yaml"),
    ("J3_joint_detach_false_splitB_seed40", "configs/_tmp_splitB_seed40_J3.yaml"),
]

for name, cfg in RUNS:
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "-m", "src.main.train_task3",
         "--config", cfg, "--model", "baseline", "--threshold_sweep"],
        cwd=str(ROOT), capture_output=False,
    )
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else "FAILED"
    print(f"  {name}: {status} in {elapsed:.1f}s")

print("\nAll splitB seed=40 runs complete.")
