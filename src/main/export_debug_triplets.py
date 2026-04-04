#!/usr/bin/env python3
"""Export one triplet per sample_id from patch_metadata.csv into debug_triplets/ (PNG)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Save debug triplet PNGs for selected sample_ids.")
    p.add_argument(
        "--metadata_csv",
        type=str,
        default="outputs/task1_topo_v3/patch_metadata.csv",
    )
    p.add_argument("--output_dir", type=str, default="debug_triplets")
    p.add_argument("--sample_ids", type=str, default="21,22,23", help="Comma-separated sample_id values.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.metadata_csv)
    if not csv_path.is_absolute():
        csv_path = REPO_ROOT / csv_path
    if not csv_path.is_file():
        print(f"Missing CSV: {csv_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    want = [s.strip() for s in args.sample_ids.split(",") if s.strip()]
    df = pd.read_csv(csv_path)
    if "sample_id" not in df.columns:
        print("CSV missing sample_id", file=sys.stderr)
        sys.exit(1)

    print("=== debug_triplets export ===\n")
    for sid in want:
        sub = df[df["sample_id"].astype(str) == str(sid)]
        if len(sub) == 0:
            print(f"No rows for sample_id={sid}", file=sys.stderr)
            sys.exit(1)
        row = sub.iloc[0]
        img_p = Path(str(row["image_path"]))
        msk_p = Path(str(row["mask_path"]))
        fov_p = Path(str(row["fov_path"]))

        for label, src in (("image", img_p), ("mask", msk_p), ("fov", fov_p)):
            if not src.is_file():
                print(f"Missing source for sample_id={sid} ({label}): {src}", file=sys.stderr)
                sys.exit(1)

        img_rgb = Image.open(img_p).convert("RGB")
        img_m = Image.open(msk_p).convert("L")
        img_f = Image.open(fov_p).convert("L")

        out_img = out_dir / f"{sid}_image.png"
        out_msk = out_dir / f"{sid}_mask.png"
        out_fov = out_dir / f"{sid}_fov.png"

        img_rgb.save(out_img)
        img_m.save(out_msk)
        img_f.save(out_fov)

        print(f"sample_id={sid}")
        print(f"  -> {out_img.resolve()}  (from {img_p.resolve()})")
        print(f"  -> {out_msk.resolve()}  (from {msk_p.resolve()})")
        print(f"  -> {out_fov.resolve()}  (from {fov_p.resolve()})")
        print()

    print(f"Done. Output directory: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
