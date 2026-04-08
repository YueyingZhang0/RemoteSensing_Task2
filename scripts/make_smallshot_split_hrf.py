#!/usr/bin/env python3
"""HRF/all small-shot split: train=10, val=5, test=remainder (45 images total).

IDs are image stems matching HRF/all/images (e.g. 01_dr, 02_g).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _all_stems(hrf_all: Path) -> list[str]:
    img_dir = hrf_all / "images"
    stems: set[str] = set()
    for p in img_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".gif"):
            continue
        stems.add(p.stem)
    return sorted(stems)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hrf_root", type=Path, default=ROOT / "HRF" / "all")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_train", type=int, default=10)
    ap.add_argument("--n_val", type=int, default=5)
    ap.add_argument("--out_dir", type=Path, required=True)
    args = ap.parse_args()

    hrf_all = args.hrf_root if args.hrf_root.is_absolute() else (ROOT / args.hrf_root).resolve()
    out_dir = args.out_dir if args.out_dir.is_absolute() else (ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_ids = _all_stems(hrf_all)
    n = len(all_ids)
    if n < args.n_train + args.n_val:
        raise ValueError(f"Not enough images: have {n}, need train+val={args.n_train + args.n_val}")

    rng = np.random.default_rng(int(args.seed))
    perm = np.array(all_ids)
    rng.shuffle(perm)
    train = sorted(perm[: args.n_train].tolist())
    val = sorted(perm[args.n_train : args.n_train + args.n_val].tolist())
    test = sorted(perm[args.n_train + args.n_val :].tolist())

    meta = {
        "dataset": "HRF/all",
        "gt": "manual1",
        "fov": "mask/*_mask.tif",
        "seed": int(args.seed),
        "n_total": n,
    }

    def dump(name: str, ids: list[str], split: str) -> None:
        (out_dir / name).write_text(
            json.dumps({"ids": ids, **meta, "split": split}, indent=2),
            encoding="utf-8",
        )

    dump("train_split_ids.json", train, "train")
    dump("val_split_ids.json", val, "val")
    dump("test_split_ids.json", test, "test")
    print(f"Wrote {out_dir} (train={len(train)} val={len(val)} test={len(test)})")


if __name__ == "__main__":
    main()
