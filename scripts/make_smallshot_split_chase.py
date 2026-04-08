#!/usr/bin/env python3
"""Create CHASE_DB1 small-shot split with subject-level grouping (L/R together).

Default (per instruction):
  - train: 8 images  (4 subjects)
  - val:   4 images  (2 subjects)
  - test:  16 images (8 subjects)

Outputs:
  - train_split_ids.json
  - val_split_ids.json
  - test_split_ids.json

IDs are CHASE stems like 'Image_01L', 'Image_01R'.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _discover_subjects(chase_dir: Path) -> Dict[str, List[str]]:
    pats = sorted(chase_dir.glob("Image_*.jpg"))
    by_subj: Dict[str, List[str]] = {}
    rx = re.compile(r"^Image_(\d{2})([LR])$")
    for p in pats:
        m = rx.match(p.stem)
        if not m:
            continue
        subj = m.group(1)
        by_subj.setdefault(subj, []).append(p.stem)
    # ensure both sides exist
    for subj, ids in list(by_subj.items()):
        if set(ids) != {f"Image_{subj}L", f"Image_{subj}R"}:
            raise ValueError(f"Subject {subj} does not have L/R pair: {ids}")
        by_subj[subj] = sorted(ids)
    if len(by_subj) != 14:
        raise ValueError(f"Expected 14 subjects in CHASE_DB1, found {len(by_subj)}")
    return by_subj


def _write(p: Path, ids: List[str], meta: Dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    obj = {"ids": ids, **meta}
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Make CHASE_DB1 subject-level small-shot split.")
    ap.add_argument("--chase_dir", type=Path, default=ROOT / "CHASE_DB1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_train_images", type=int, default=8)
    ap.add_argument("--n_val_images", type=int, default=4)
    ap.add_argument("--out_dir", type=Path, required=True)
    args = ap.parse_args()

    chase_dir = args.chase_dir if args.chase_dir.is_absolute() else (ROOT / args.chase_dir).resolve()
    out_dir = args.out_dir if args.out_dir.is_absolute() else (ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.n_train_images % 2 != 0 or args.n_val_images % 2 != 0:
        raise ValueError("n_train_images and n_val_images must be even (subject pairs)")
    n_train_subj = args.n_train_images // 2
    n_val_subj = args.n_val_images // 2

    by_subj = _discover_subjects(chase_dir)
    subjects = sorted(by_subj.keys())
    rng = np.random.default_rng(int(args.seed))
    perm = subjects.copy()
    rng.shuffle(perm)

    train_subj = sorted(perm[:n_train_subj])
    val_subj = sorted(perm[n_train_subj : n_train_subj + n_val_subj])
    test_subj = sorted(perm[n_train_subj + n_val_subj :])

    def flatten(subjs: List[str]) -> List[str]:
        ids: List[str] = []
        for s in subjs:
            ids.extend(by_subj[s])
        return ids

    train_ids = flatten(train_subj)
    val_ids = flatten(val_subj)
    test_ids = flatten(test_subj)

    meta = {
        "dataset": "CHASE_DB1",
        "grouping": "subject_id (Image_XX)",
        "seed": int(args.seed),
        "n_subjects": len(subjects),
        "train_subjects": train_subj,
        "val_subjects": val_subj,
        "test_subjects": test_subj,
    }

    _write(out_dir / "train_split_ids.json", train_ids, {**meta, "split": "train"})
    _write(out_dir / "val_split_ids.json", val_ids, {**meta, "split": "val"})
    _write(out_dir / "test_split_ids.json", test_ids, {**meta, "split": "test"})
    print(f"Wrote split to {out_dir}")


if __name__ == "__main__":
    main()

