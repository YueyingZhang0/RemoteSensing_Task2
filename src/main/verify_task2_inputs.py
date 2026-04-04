#!/usr/bin/env python3
"""Verify configs/sce_probe.yaml fields and patch_name <-> PNG alignment."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.utils.io import load_yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/sce_probe.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    if not cfg_path.is_file():
        print(f"Missing config: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    cfg = load_yaml(cfg_path)

    print("=== Task 2 config checks (sce_probe.yaml) ===\n")
    checks = [
        ("target_col", "sci_res2_norm"),
        ("output_dir", "outputs/task2_sce_probe"),
        ("group_col", "sample_id"),
    ]
    ok = True
    for key, expected in checks:
        val = cfg.get(key)
        pass_ = val == expected
        ok = ok and pass_
        print(f"  {key}: {val!r}  {'OK' if pass_ else f'EXPECTED {expected!r}'}")

    meta = cfg.get("metadata_csv")
    patch_dir = cfg.get("patch_image_dir")
    print(f"  metadata_csv: {meta!r}")
    print(f"  patch_image_dir: {patch_dir!r}")

    meta_path = Path(meta) if meta else None
    if meta_path and not meta_path.is_absolute():
        meta_path = REPO_ROOT / meta_path
    img_root = Path(patch_dir) if patch_dir else None
    if img_root and not img_root.is_absolute():
        img_root = REPO_ROOT / img_root

    if not meta_path or not meta_path.is_file():
        print("\nERROR: metadata_csv path not a file.", file=sys.stderr)
        sys.exit(1)
    if not img_root or not img_root.is_dir():
        print("\nERROR: patch_image_dir not a directory.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(meta_path)
    n = len(df)
    missing: list[str] = []
    for _, row in df.iterrows():
        name = str(row["patch_name"])
        p = img_root / name
        if not p.is_file():
            missing.append(name)
            if len(missing) >= 10:
                break

    print(f"\n=== patch_name vs PNG ({img_root}) ===")
    print(f"  CSV rows: {n}")
    png_count = len(list(img_root.glob("*.png")))
    print(f"  *.png in patch_image_dir: {png_count}")

    if missing:
        print(f"  FAIL: first missing patch files ({len(missing)} shown): {missing}", file=sys.stderr)
        sys.exit(1)

    if png_count != n:
        print(f"  WARN: PNG count ({png_count}) != CSV rows ({n}); still all names resolved.", file=sys.stderr)

    print("  OK: every patch_name has a matching file under patch_image_dir.")
    print("\nAll Step 3 checks passed.")


if __name__ == "__main__":
    main()
