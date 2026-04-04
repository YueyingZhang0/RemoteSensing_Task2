#!/usr/bin/env python3
"""Compare config_used.yaml between two Task 3 run directories (weight_mode, output_dir, timestamps)."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def _load_config_used(d: Path) -> dict:
    p = d / "config_used.yaml"
    if not p.is_file():
        raise FileNotFoundError(f"Missing config_used.yaml in {d}")
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    p = argparse.ArgumentParser(description="Verify two run dirs have distinct config_used and sane weight_mode.")
    p.add_argument("dir_a", type=str, help="First run directory (e.g. oracle_linear)")
    p.add_argument("dir_b", type=str, help="Second run directory (e.g. oracle_smart)")
    p.add_argument("--root", type=str, default=".", help="Project root if paths are relative")
    args = p.parse_args()
    root = Path(args.root).resolve()
    da = Path(args.dir_a)
    db = Path(args.dir_b)
    if not da.is_absolute():
        da = root / da
    if not db.is_absolute():
        db = root / db
    ca = _load_config_used(da)
    cb = _load_config_used(db)

    oa = ca.get("output_dir")
    ob = cb.get("output_dir")
    wa = (ca.get("difficulty_weighting") or {}).get("weight_mode")
    wb = (cb.get("difficulty_weighting") or {}).get("weight_mode")
    ta = ca.get("timestamp")
    tb = cb.get("timestamp")

    print(f"dir_a: {da.resolve()}")
    print(f"  output_dir: {oa}")
    print(f"  difficulty_weighting.weight_mode: {wa}")
    print(f"  timestamp: {ta}")
    print(f"dir_b: {db.resolve()}")
    print(f"  output_dir: {ob}")
    print(f"  difficulty_weighting.weight_mode: {wb}")
    print(f"  timestamp: {tb}")

    issues: list[str] = []
    if oa and ob and Path(oa).resolve() == Path(ob).resolve():
        issues.append("output_dir resolves to the same path — runs may overwrite each other.")
    dwa = ca.get("difficulty_weighting") or {}
    dwb = cb.get("difficulty_weighting") or {}
    if dwa.get("enabled") and dwb.get("enabled") and wa is not None and wb is not None and wa == wb:
        issues.append("weight_mode is identical while both have difficulty on — expected different modes for ablation.")
    if str(da.resolve()) == str(db.resolve()):
        issues.append("Same directory passed twice.")

    if issues:
        print("\nWARNINGS:")
        for x in issues:
            print(f"  - {x}")
        raise SystemExit(1)
    print("\nOK: distinct output_dir and weight_mode fields present.")


if __name__ == "__main__":
    main()
