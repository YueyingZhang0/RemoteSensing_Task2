#!/usr/bin/env python3
"""Run nnU-Net v2 inference on exported Task3 imagesTr (all cases) with probabilities.

Requires finished training (e.g. checkpoint_final.pth in fold_0). Sets nnUNet_* paths from repo root.

  python scripts/nnunetv2_predict_task3.py --out nnUNet_predictions/task3_fold0

Then evaluate val-only metrics with eval_nnunet_task3_predictions.py (it aligns by case_id and uses val rows only).

Uses predict_entry_point_modelfolder (no dependency on nnUNet_results layout for discovery beyond -m).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-id", type=int, default=503)
    ap.add_argument(
        "--out",
        type=str,
        default="nnUNet_predictions/task3_fold0_prob",
        help="Output folder for .png and .npz",
    )
    ap.add_argument("--fold", type=str, default="0")
    ap.add_argument(
        "--chk",
        type=str,
        default="checkpoint_final.pth",
        help="checkpoint_final.pth after training completes; use checkpoint_best.pth if needed",
    )
    ap.add_argument("--device", type=str, default="cuda", choices=("cuda", "cpu", "mps"))
    ap.add_argument("--no-probabilities", action="store_true", help="Omit .npz (not recommended for threshold sweep)")
    args = ap.parse_args()

    root = _ROOT
    os_raw = root / "nnUNet_raw"
    os_pre = root / "nnUNet_preprocessed"
    os_res = root / "nnUNet_results"

    os.environ.setdefault("nnUNet_raw", str(os_raw.resolve()))
    os.environ.setdefault("nnUNet_preprocessed", str(os_pre.resolve()))
    os.environ.setdefault("nnUNet_results", str(os_res.resolve()))

    ds = f"Dataset{int(args.dataset_id):03d}_TASK3"
    input_dir = os_raw / ds / "imagesTr"
    model_dir = os_res / ds / "nnUNetTrainer__nnUNetPlans__2d"
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Missing {input_dir} (run export_task3_for_nnunet_v2 first)")
    fold_dir = model_dir / f"fold_{args.fold}"
    ck = fold_dir / args.chk
    if not ck.is_file():
        raise FileNotFoundError(
            f"Missing {ck}. Wait for training to finish or pass --chk checkpoint_best.pth / checkpoint_latest.pth"
        )

    argv = [
        "nnUNetv2_predict_from_modelfolder",
        "-i",
        str(input_dir.resolve()),
        "-o",
        str(out_dir.resolve()),
        "-m",
        str(model_dir.resolve()),
        "-f",
        args.fold,
        "-chk",
        args.chk,
        "-device",
        args.device,
    ]
    if not args.no_probabilities:
        argv.append("--save_probabilities")

    old = sys.argv
    try:
        sys.argv = argv
        from nnunetv2.inference.predict_from_raw_data import predict_entry_point_modelfolder

        predict_entry_point_modelfolder()
    finally:
        sys.argv = old


if __name__ == "__main__":
    main()
