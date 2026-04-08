#!/usr/bin/env python3
"""1-image smoke tests before full CHASE zero-shot matrix (per user protocol).

Torch models (P0 / J3 / Attention / Swin): same ``eval_external_thinvessel`` sliding-window path, ``--max_images 1``.

nnU-Net:
  1) Export one case with ``export_chase_for_nnunet_inference.py --max_cases 1``.
  2) You run ``nnunetv2_predict_task3.py --input-dir ...`` (or pass ``--run_nnunet_predict`` if nnunetv2 + checkpoint exist).
  3) ``eval_nnunet_chase_thinvessel.py --max_images 1``.

If a backbone fails here, stop and fix the environment/checkpoint before full runs (avoid large refactors).

Example:
  python scripts/smoke_chase_zeroshot_backbones.py --split_ids_json outputs/ext_chase_split_seed42/test_split_ids.json
  python scripts/smoke_chase_zeroshot_backbones.py --only nnunet --run_nnunet_predict
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_ids_json", type=Path, default=Path("outputs/ext_chase_split_seed42/test_split_ids.json"))
    ap.add_argument(
        "--only",
        type=str,
        choices=("all", "torch", "nnunet"),
        default="all",
    )
    ap.add_argument(
        "--run_nnunet_predict",
        action="store_true",
        help="After export, invoke nnunetv2_predict_task3.py (requires trained fold + nnunetv2).",
    )
    args = ap.parse_args()

    sp = args.split_ids_json if args.split_ids_json.is_absolute() else ROOT / args.split_ids_json

    if args.only in ("all", "torch"):
        torch_runs = [
            (
                "outputs/_smoke/chase_p0",
                "outputs/task3_v2_1_baseline_same_split/best_model.pt",
                "baseline",
            ),
            (
                "outputs/_smoke/chase_j3",
                "outputs/task3_J3_joint_detach_false/best_model.pt",
                "joint",
            ),
            (
                "outputs/_smoke/chase_attention",
                "outputs/task3_attention_unet_baseline/best_model.pt",
                "attention_unet",
            ),
            (
                "outputs/_smoke/chase_swin",
                "outputs/task3_swin_unet_baseline/best_model.pt",
                "swin_unet",
            ),
        ]
        for out_dir, ckpt, kind in torch_runs:
            ck = ROOT / ckpt
            if not ck.is_file():
                print(f"SKIP torch smoke ({kind}): missing checkpoint {ck}", flush=True)
                continue
            od = ROOT / out_dir
            if od.exists():
                for p in od.glob("*"):
                    p.unlink()
            else:
                od.mkdir(parents=True, exist_ok=True)
            _run(
                [
                    sys.executable,
                    "scripts/eval_external_thinvessel.py",
                    "--dataset",
                    "chase_db1_1st",
                    "--ckpt",
                    str(ck),
                    "--kind",
                    kind,
                    "--out_dir",
                    str(out_dir),
                    "--split_ids_json",
                    str(sp),
                    "--max_images",
                    "1",
                ]
            )

    if args.only in ("all", "nnunet"):
        export_dir = ROOT / "nnUNet_inference_inputs" / "_smoke_chase_one"
        if export_dir.exists():
            for p in export_dir.glob("*"):
                if p.is_file():
                    p.unlink()
        export_dir.mkdir(parents=True, exist_ok=True)
        _run(
            [
                sys.executable,
                "scripts/export_chase_for_nnunet_inference.py",
                "--split_ids_json",
                str(sp),
                "--out",
                str(export_dir.relative_to(ROOT)),
                "--max_cases",
                "1",
            ]
        )
        pred_dir = ROOT / "nnUNet_predictions" / "_smoke_chase_one"
        print(
            f"\n[nnU-Net] Exported 1 case to {export_dir}. "
            f"If not using --run_nnunet_predict, run:\n"
            f"  python scripts/nnunetv2_predict_task3.py "
            f"--input-dir {export_dir.relative_to(ROOT)} --out {pred_dir.relative_to(ROOT)}\n",
            flush=True,
        )
        if args.run_nnunet_predict:
            pred_dir.mkdir(parents=True, exist_ok=True)
            for p in pred_dir.glob("*"):
                if p.is_file():
                    p.unlink()
            _run(
                [
                    sys.executable,
                    "scripts/nnunetv2_predict_task3.py",
                    "--input-dir",
                    str(export_dir.relative_to(ROOT)),
                    "--out",
                    str(pred_dir.relative_to(ROOT)),
                ]
            )
            smoke_out = ROOT / "outputs" / "_smoke" / "chase_nnunet"
            if smoke_out.exists():
                for p in smoke_out.glob("*"):
                    p.unlink()
            else:
                smoke_out.mkdir(parents=True, exist_ok=True)
            _run(
                [
                    sys.executable,
                    "scripts/eval_nnunet_chase_thinvessel.py",
                    "--pred-dir",
                    str(pred_dir.relative_to(ROOT)),
                    "--split_ids_json",
                    str(sp),
                    "--out_dir",
                    str(smoke_out.relative_to(ROOT)),
                    "--max_images",
                    "1",
                ]
            )

    print("Smoke flow finished.", flush=True)


if __name__ == "__main__":
    main()
