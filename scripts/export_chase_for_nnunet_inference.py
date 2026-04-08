#!/usr/bin/env python3
"""Export CHASE_DB1 RGB cases to an nnU-Net v2 **inference** input folder (3-channel PNG layout).

Matches Task3 export convention: ``{case_id}_0000.png`` … ``_0002.png`` per case so a model trained
on Dataset503_TASK3 can run ``nnUNetv2_predict_from_modelfolder`` with ``--input-dir`` here.

Does not touch canonical Task3 exports.

Example:
  python scripts/export_chase_for_nnunet_inference.py \\
    --split_ids_json outputs/ext_chase_split_seed42/test_split_ids.json \\
    --out nnUNet_inference_inputs/chase_zeroshot_eval

Then:
  python scripts/nnunetv2_predict_task3.py \\
    --input-dir nnUNet_inference_inputs/chase_zeroshot_eval \\
    --out nnUNet_predictions/chase_zeroshot_prob
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description="Export CHASE_DB1 images for nnU-Net v2 predict -i.")
    ap.add_argument("--split_ids_json", type=Path, required=True)
    ap.add_argument(
        "--chase_dir",
        type=Path,
        default=None,
        help="CHASE_DB1 root (default: <repo>/CHASE_DB1)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output folder (will contain *_0000.png … per case).",
    )
    ap.add_argument(
        "--max_cases",
        type=int,
        default=0,
        help="If >0, export only the first N ids (smoke test).",
    )
    args = ap.parse_args()

    chase = args.chase_dir if args.chase_dir is not None else ROOT / "CHASE_DB1"
    chase = chase.resolve() if not chase.is_absolute() else chase
    if not chase.is_dir():
        raise FileNotFoundError(chase)

    sp = args.split_ids_json if args.split_ids_json.is_absolute() else (ROOT / args.split_ids_json).resolve()
    ids = [str(x) for x in json.loads(sp.read_text(encoding="utf-8")).get("ids", [])]
    if int(args.max_cases) > 0:
        ids = ids[: int(args.max_cases)]

    out = args.out if args.out.is_absolute() else (ROOT / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    for cid in ids:
        ip = chase / f"{cid}.jpg"
        if not ip.is_file():
            raise FileNotFoundError(ip)
        rgb = np.asarray(Image.open(ip).convert("RGB"), dtype=np.uint8)
        for c in range(3):
            ch = rgb[:, :, c]
            op = out / f"{cid}_{c:04d}.png"
            Image.fromarray(ch, mode="L").save(op)

    meta = {
        "source": "CHASE_DB1",
        "split_ids_json": str(sp),
        "n_cases": len(ids),
        "case_ids": ids,
        "nnunet_note": "nnU-Net uses its own preprocessing/resampling at inference; declare in paper table footnote.",
    }
    (out / "chase_nnunet_export_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {len(ids) * 3} PNGs under {out}")


if __name__ == "__main__":
    main()
