#!/usr/bin/env python3
"""Compare Task 3 baseline vs SCE v1 vs v2 oracle vs v3; optional multi-run aggregation."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from src.utils.io import load_json, save_json


def _find_val_metrics(d: Path) -> Path | None:
    for name in ("val_metrics.json", "Baseline_val_metrics.json", "baseline_val_metrics.json"):
        p = d / name
        if p.is_file():
            return p
    return None


def load_task3_run_metrics(d: Path) -> dict:
    vp = _find_val_metrics(d)
    if vp is None:
        raise FileNotFoundError(f"No val_metrics in {d}")
    vm = load_json(vp)
    hm_path = d / "hard_patch_metrics.json"
    hm = load_json(hm_path) if hm_path.is_file() else {}
    tm_path = d / "threshold_metrics.json"
    tm = load_json(tm_path) if tm_path.is_file() else {}
    out = {
        "val_dice_mean": vm.get("val_dice_mean"),
        "val_recall_mean": vm.get("val_recall_mean"),
        "hard_dice_mean": vm.get("hard_dice_mean", hm.get("hard_dice_mean")),
        "hard_recall_mean": vm.get("hard_recall_mean", hm.get("hard_recall_mean")),
        "val_sci_corr": vm.get("val_sci_corr"),
        "loss_mode": vm.get("loss_mode"),
        "best_val_dice_threshold": tm.get("best_val_dice_threshold"),
        "best_hard_dice_threshold": tm.get("best_hard_dice_threshold"),
    }
    return out


def _load_metrics(d: Path) -> dict:
    return load_task3_run_metrics(d)


def aggregate_task3_runs(directories: list[Path]) -> dict:
    """Mean/std over val_dice_mean, val_recall_mean, hard_dice_mean, hard_recall_mean."""
    keys = ("val_dice_mean", "val_recall_mean", "hard_dice_mean", "hard_recall_mean")
    rows: list[dict] = []
    for d in directories:
        d = d.resolve()
        try:
            m = load_task3_run_metrics(d)
            rows.append({"dir": str(d), **m})
        except FileNotFoundError:
            rows.append({"dir": str(d), "error": "missing val_metrics"})

    collected: dict[str, list[float]] = {k: [] for k in keys}
    for r in rows:
        for k in keys:
            v = r.get(k)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            collected[k].append(float(v))

    agg: dict[str, dict] = {}
    for k, arr in collected.items():
        if not arr:
            agg[k] = {"mean": None, "std": None, "n": 0}
            continue
        a = np.asarray(arr, dtype=np.float64)
        std = float(a.std(ddof=1)) if len(a) > 1 else 0.0
        agg[k] = {"mean": float(a.mean()), "std": std, "n": int(len(a))}

    return {"runs": rows, "aggregate": agg}


def default_v2_1_run_dirs() -> list[tuple[str, str]]:
    return [
        ("baseline_same_split", "outputs/task3_v2_1_baseline_same_split"),
        ("oracle_linear_same_split", "outputs/task3_v2_1_oracle_linear_same_split"),
        ("oracle_smart_same_split", "outputs/task3_v2_1_oracle_smart_same_split"),
        ("oracle_linear_sampler_same_split", "outputs/task3_v2_1_oracle_linear_sampler_same_split"),
        ("oracle_smart_sampler_same_split", "outputs/task3_v2_1_oracle_smart_sampler_same_split"),
    ]


def build_v2_1_comparison_table(root: Path) -> dict:
    rows_out: list[dict] = []
    for variant, rel in default_v2_1_run_dirs():
        d = root / rel
        try:
            m = load_task3_run_metrics(d)
            rows_out.append(
                {
                    "variant": variant,
                    "dir": str(d.resolve()),
                    "val_dice_mean": m.get("val_dice_mean"),
                    "val_recall_mean": m.get("val_recall_mean"),
                    "hard_dice_mean": m.get("hard_dice_mean"),
                    "hard_recall_mean": m.get("hard_recall_mean"),
                    "best_val_dice_threshold": m.get("best_val_dice_threshold"),
                    "best_hard_dice_threshold": m.get("best_hard_dice_threshold"),
                    "loss_mode": m.get("loss_mode"),
                }
            )
        except FileNotFoundError:
            rows_out.append({"variant": variant, "dir": str(d.resolve()), "error": "missing val_metrics"})
    return {"v2_1_runs": rows_out, "split_note": "All runs must use the same split_json (see each config_used.yaml)."}


def write_v2_1_csv(table: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    runs = table.get("v2_1_runs") or []
    fields = [
        "variant",
        "val_dice_mean",
        "val_recall_mean",
        "hard_dice_mean",
        "hard_recall_mean",
        "best_val_dice_threshold",
        "best_hard_dice_threshold",
        "loss_mode",
        "dir",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in runs:
            w.writerow({k: r.get(k) for k in fields})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline_dir", type=str, default="outputs/task3_unet_baseline")
    p.add_argument("--sce_dir", type=str, default="outputs/task3_unet_sce")
    p.add_argument("--v2_dir", type=str, default="outputs/task3_v2_oracle_weighted")
    p.add_argument("--v3_frozen_dir", type=str, default=None, help="Optional v3 frozen-probe run directory.")
    p.add_argument("--v3_joint_dir", type=str, default=None, help="Optional v3 joint-difficulty run directory.")
    p.add_argument("--out", type=str, default="outputs/task3_compare_summary.json")
    p.add_argument("--root", type=str, default=".")
    p.add_argument(
        "--aggregate_dirs",
        type=str,
        default=None,
        help="Comma-separated run directories; writes mean/std over metrics to --aggregate_out.",
    )
    p.add_argument(
        "--aggregate_out",
        type=str,
        default="outputs/task3_aggregate_runs.json",
    )
    p.add_argument(
        "--v2_1_table",
        action="store_true",
        help="Build v2.1 comparison table from default output dirs under --root.",
    )
    p.add_argument("--v2_1_out_json", type=str, default="outputs/task3_v2_1_experiments/comparison_summary.json")
    p.add_argument("--v2_1_out_csv", type=str, default="outputs/task3_v2_1_experiments/comparison_summary.csv")
    p.add_argument("--v2_1_smart_dir", type=str, default=None)
    p.add_argument("--v2_1_smart_sampler_dir", type=str, default=None)
    p.add_argument("--v2_1_linear_sampler_dir", type=str, default=None)
    args = p.parse_args()
    root = Path(args.root).resolve()

    def R(s: str) -> Path:
        x = Path(s)
        return x if x.is_absolute() else root / x

    if args.v2_1_table:
        tab = build_v2_1_comparison_table(root)
        jpath = R(args.v2_1_out_json)
        cpath = R(args.v2_1_out_csv)
        jpath.parent.mkdir(parents=True, exist_ok=True)
        save_json(tab, jpath)
        write_v2_1_csv(tab, cpath)
        print(json.dumps(tab, indent=2))
        print(f"Wrote {jpath} and {cpath}")
        return

    if args.aggregate_dirs:
        dirs = [R(s.strip()) for s in args.aggregate_dirs.split(",") if s.strip()]
        ag = aggregate_task3_runs(dirs)
        outp = R(args.aggregate_out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        save_json(ag, outp)
        print(json.dumps(ag, indent=2))
        print(f"Wrote {outp}")
        return

    bd, sd, vd = R(args.baseline_dir), R(args.sce_dir), R(args.v2_dir)

    def safe_load(d: Path):
        try:
            return _load_metrics(d)
        except FileNotFoundError:
            return None

    summary = {
        "baseline": safe_load(bd),
        "sce_v1": safe_load(sd),
        "v2_oracle_weighted": safe_load(vd),
    }
    if args.v3_frozen_dir:
        summary["v3_frozen_probe"] = safe_load(R(args.v3_frozen_dir))
    if args.v3_joint_dir:
        summary["v3_joint_difficulty"] = safe_load(R(args.v3_joint_dir))
    if args.v2_1_smart_dir:
        summary["v2_1_oracle_smart"] = safe_load(R(args.v2_1_smart_dir))
    if args.v2_1_smart_sampler_dir:
        summary["v2_1_oracle_smart_sampler"] = safe_load(R(args.v2_1_smart_sampler_dir))
    if args.v2_1_linear_sampler_dir:
        summary["v2_1_oracle_linear_sampler"] = safe_load(R(args.v2_1_linear_sampler_dir))

    vb, v2 = summary["baseline"], summary["v2_oracle_weighted"]
    if vb and v2:
        eps = 0.003
        summary["v2_vs_baseline"] = {
            "hard_dice_delta": (v2["hard_dice_mean"] or 0) - (vb["hard_dice_mean"] or 0),
            "hard_recall_delta": (v2["hard_recall_mean"] or 0) - (vb["hard_recall_mean"] or 0),
            "val_dice_delta": (v2["val_dice_mean"] or 0) - (vb["val_dice_mean"] or 0),
            "pass_min_hard_dice": (v2["hard_dice_mean"] or 0) >= (vb["hard_dice_mean"] or 0) - eps,
            "pass_min_val_dice": (v2["val_dice_mean"] or 0) >= (vb["val_dice_mean"] or 0) - eps,
            "pass_hard_recall": (v2["hard_recall_mean"] or 0) >= (vb["hard_recall_mean"] or 0),
        }
    else:
        summary["v2_vs_baseline"] = None

    outp = R(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    save_json(summary, outp)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {outp}")

    csvp = outp.with_suffix(".csv")
    rows = []
    for key in (
        "baseline",
        "sce_v1",
        "v2_oracle_weighted",
        "v3_frozen_probe",
        "v3_joint_difficulty",
        "v2_1_oracle_smart",
        "v2_1_oracle_smart_sampler",
        "v2_1_oracle_linear_sampler",
    ):
        if key not in summary or summary[key] is None:
            continue
        m = summary[key]
        rows.append(
            {
                "variant": key,
                "val_dice_mean": m.get("val_dice_mean"),
                "val_recall_mean": m.get("val_recall_mean"),
                "hard_dice_mean": m.get("hard_dice_mean"),
                "hard_recall_mean": m.get("hard_recall_mean"),
                "best_val_dice_threshold": m.get("best_val_dice_threshold"),
                "best_hard_dice_threshold": m.get("best_hard_dice_threshold"),
            }
        )
    if rows:
        fields = [
            "variant",
            "val_dice_mean",
            "val_recall_mean",
            "hard_dice_mean",
            "hard_recall_mean",
            "best_val_dice_threshold",
            "best_hard_dice_threshold",
        ]
        with open(csvp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {csvp}")


if __name__ == "__main__":
    main()