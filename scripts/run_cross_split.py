#!/usr/bin/env python3
"""Phase 5: Run P0/P5/J3 on 2 new splits for cross-split validation."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ROOT = Path(__file__).resolve().parents[1]

SPLITS = [
    ("splitB", "outputs/task3_split/split_info_rs100.json"),
    ("splitC", "outputs/task3_split/split_info_rs200.json"),
]

METHODS = [
    ("P0_baseline", "configs/task3_v2_1_baseline.yaml"),
    ("P5_oracle_matched_trigger", "configs/task3_v3_1_oracle_matched_trigger.yaml"),
    ("J3_joint_detach_false", "configs/task3_J3_joint_detach_false.yaml"),
]


def run_one(name: str, cfg: str, out_dir: str, split_json: str) -> dict:
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    import yaml
    cfg_data = yaml.safe_load(Path(ROOT / cfg).read_text(encoding="utf-8"))
    cfg_data["split_json"] = split_json
    cfg_data["output_dir"] = out_dir
    if "output_dir_baseline" in cfg_data:
        cfg_data["output_dir_baseline"] = out_dir

    tmp_cfg = ROOT / "configs" / f"_tmp_cross_split.yaml"
    with open(tmp_cfg, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_data, f, sort_keys=False, allow_unicode=True)

    cmd = [
        sys.executable, "-m", "src.main.train_task3",
        "--config", str(tmp_cfg),
        "--model", "baseline",
        "--threshold_sweep",
    ]

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=False)
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else "FAILED"
    print(f"  {name}: {status} in {elapsed:.1f}s")

    if tmp_cfg.exists():
        tmp_cfg.unlink()

    return {"name": name, "exit_code": result.returncode, "elapsed_s": round(elapsed, 1)}


def main():
    results = []
    for split_name, split_json in SPLITS:
        for method_name, cfg in METHODS:
            run_name = f"{method_name}_{split_name}"
            out_dir = f"outputs/task3_{method_name}_{split_name}"
            info = run_one(run_name, cfg, out_dir, split_json)
            results.append(info)
            if info["exit_code"] != 0:
                print(f"  WARNING: {run_name} FAILED!")

    print(f"\n{'='*60}")
    print("  CROSS-SPLIT RUNS COMPLETE")
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
