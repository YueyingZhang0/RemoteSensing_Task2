#!/usr/bin/env python3
"""Set nnU-Net v2 folder env vars from repo root and run training (dataset 503, 2d, fold 0).

Avoids relying on PATH for nnUNetv2_train.exe.

  python scripts/nnunetv2_train_task3.py
  python scripts/nnunetv2_train_task3.py --device cpu
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="nnU-Net v2 train Task3 export (503, 2d, fold 0).")
    ap.add_argument("--dataset-id", type=int, default=503)
    ap.add_argument("--device", type=str, default="cuda", choices=("cuda", "cpu", "mps"))
    ap.add_argument("--dry-run", action="store_true", help="Only run pre-flight checks; do not start training.")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    raw = root / "nnUNet_raw"
    pre = root / "nnUNet_preprocessed"
    res = root / "nnUNet_results"

    os.environ["nnUNet_raw"] = str(raw.resolve())
    os.environ["nnUNet_preprocessed"] = str(pre.resolve())
    os.environ["nnUNet_results"] = str(res.resolve())

    ds_name = f"Dataset{int(args.dataset_id):03d}_TASK3"
    raw_ds = raw / ds_name
    ds_pre = pre / ds_name
    plans = ds_pre / "nnUNetPlans.json"

    checks = {
        "raw_dataset_dir": raw_ds.is_dir(),
        "preprocessed_dir": ds_pre.is_dir(),
        "nnUNetPlans_json": plans.is_file(),
        "dataset_json": (ds_pre / "dataset.json").is_file(),
    }

    if not checks["nnUNetPlans_json"]:
        print(
            "Missing preprocessed plans. From repo root run:\n"
            "  python -m nnunetv2.experiment_planning.plan_and_preprocess_entrypoints "
            f"-d {args.dataset_id} -c 2d --verify_dataset_integrity\n"
            "(Set nnUNet_* env vars first if that command cannot find the dataset.)"
        )
        sys.exit(2)

    res.mkdir(parents=True, exist_ok=True)
    try:
        test_write = res / ".write_test"
        test_write.write_text("ok", encoding="utf-8")
        test_write.unlink(missing_ok=True)
        writable = True
    except OSError:
        writable = False

    if not writable:
        print(f"Cannot write to nnUNet_results: {res}")
        sys.exit(3)

    try:
        import nnunetv2  # noqa: F401
    except ImportError:
        print("nnunetv2 is not installed for this Python. Try: python -m pip install nnunetv2")
        sys.exit(4)

    cmd = [
        sys.executable,
        "-m",
        "nnunetv2.run.run_training",
        str(args.dataset_id),
        "2d",
        "0",
        "-device",
        args.device,
    ]

    if args.dry_run:
        print("Dry-run OK. Would run:", " ".join(cmd))
        return

    print("Starting nnU-Net training (long run). Checkpoints under nnUNet_results.")
    r = subprocess.run(cmd, cwd=str(root), env=os.environ.copy())
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
