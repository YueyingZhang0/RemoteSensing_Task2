#!/usr/bin/env python3
"""Summarize Task 3 v2.1 minimal matrix: metrics @ 0.5, best thresholds, CSV + JSON."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _th_key(t: Optional[float]) -> Optional[str]:
    if t is None:
        return None
    return f"{float(t):.2f}"


def load_row(run_id: str, out_dir: Path) -> Dict[str, Any]:
    vm_path = out_dir / "val_metrics.json"
    tm_path = out_dir / "threshold_metrics.json"
    if not vm_path.is_file():
        raise FileNotFoundError(vm_path)
    if not tm_path.is_file():
        raise FileNotFoundError(tm_path)
    vm = json.loads(vm_path.read_text(encoding="utf-8"))
    tm = json.loads(tm_path.read_text(encoding="utf-8"))
    per = tm.get("per_threshold") or {}
    t05 = per.get("0.50") or {}
    bht = tm.get("best_hard_dice_threshold")
    bvt = tm.get("best_val_dice_threshold")
    kh = _th_key(bht)
    kv = _th_key(bvt)
    row: Dict[str, Any] = {
        "run_id": run_id,
        "output_dir": str(out_dir.resolve()),
        "loss_mode": vm.get("loss_mode"),
        "val_dice_mean_at_0_5": t05.get("val_dice_mean", vm.get("val_dice_mean")),
        "val_recall_mean_at_0_5": t05.get("val_recall_mean", vm.get("val_recall_mean")),
        "hard_dice_mean_at_0_5": t05.get("hard_dice_mean", vm.get("hard_dice_mean")),
        "hard_recall_mean_at_0_5": t05.get("hard_recall_mean", vm.get("hard_recall_mean")),
        "best_val_dice_threshold": bvt,
        "best_val_dice_mean": per.get(kv, {}).get("val_dice_mean") if kv else None,
        "best_hard_dice_threshold": bht,
        "primary_hard_dice_mean": per.get(kh, {}).get("hard_dice_mean") if kh else None,
        "hard_recall_at_best_hard_dice_threshold": per.get(kh, {}).get("hard_recall_mean") if kh else None,
        "val_dice_mean_at_best_val_dice_threshold": per.get(kv, {}).get("val_dice_mean") if kv else None,
    }
    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=".", help="Project root")
    p.add_argument("--out_json", type=str, default="outputs/task3_v2_1_minimal_matrix_summary.json")
    p.add_argument("--out_csv", type=str, default="outputs/task3_v2_1_minimal_matrix_summary.csv")
    args = p.parse_args()
    root = Path(args.root).resolve()

    runs: List[tuple[str, Path]] = [
        ("P0_baseline", root / "outputs" / "task3_v2_1_baseline_same_split"),
        ("P1_oracle_true_linear", root / "outputs" / "task3_v2_1_oracle_true_linear_same_split"),
        ("P2_oracle_quadratic", root / "outputs" / "task3_v2_1_oracle_quadratic_same_split"),
        ("P3_best_plus_sampler", root / "outputs" / "task3_v2_1_best_plus_sampler_same_split"),
    ]
    rows = [load_row(rid, d) for rid, d in runs]

    def _fget(r: Dict[str, Any], k: str) -> float:
        v = r.get(k)
        return float(v) if v is not None and not (isinstance(v, float) and v != v) else float("nan")

    base = rows[0]
    p1, p2, p3 = rows[1], rows[2], rows[3]
    d_tl = _fget(p1, "primary_hard_dice_mean") - _fget(base, "primary_hard_dice_mean")
    d_quad = _fget(p2, "primary_hard_dice_mean") - _fget(base, "primary_hard_dice_mean")
    winner = "oracle_quadratic" if _fget(p2, "primary_hard_dice_mean") >= _fget(p1, "primary_hard_dice_mean") else "oracle_true_linear"
    d_p3_vs_winner = _fget(p3, "primary_hard_dice_mean") - max(
        _fget(p1, "primary_hard_dice_mean"), _fget(p2, "primary_hard_dice_mean")
    )

    summary: Dict[str, Any] = {
        "runs": rows,
        "primary_metric": "hard_dice_mean_at_best_hard_dice_threshold",
        "answers": {
            "true_linear_better_than_baseline_on_primary": bool(d_tl >= 0.005),
            "true_linear_weak_gain_vs_baseline": bool(0.003 <= d_tl < 0.005),
            "quadratic_better_than_true_linear_on_primary": bool(
                _fget(p2, "primary_hard_dice_mean") > _fget(p1, "primary_hard_dice_mean")
            ),
            "sampler_improved_over_best_weighting_primary": bool(d_p3_vs_winner > 0),
            "p1_vs_baseline_delta_primary": d_tl,
            "p2_vs_baseline_delta_primary": d_quad,
            "p3_vs_best_p1_p2_delta_primary": d_p3_vs_winner,
            "declared_p1_p2_winner_for_p3": winner,
        },
    }
    out_j = root / args.out_json
    out_j.parent.mkdir(parents=True, exist_ok=True)
    out_j.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    keys = list(rows[0].keys())
    out_c = root / args.out_csv
    with open(out_c, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    print(
        "run_id".ljust(26),
        "primary_hard_dice".ljust(18),
        "hard_dice@0.5".ljust(14),
        "val_dice@0.5".ljust(14),
        "loss_mode",
    )
    print("-" * 96)
    for r in rows:
        ph = r.get("primary_hard_dice_mean")
        hd5 = r.get("hard_dice_mean_at_0_5")
        vd5 = r.get("val_dice_mean_at_0_5")
        lm = r.get("loss_mode") or "baseline"
        print(
            str(r["run_id"]).ljust(26),
            (f"{ph:.6f}" if ph is not None else "n/a").ljust(18),
            (f"{hd5:.6f}" if hd5 is not None else "n/a").ljust(14),
            (f"{vd5:.6f}" if vd5 is not None else "n/a").ljust(14),
            str(lm),
        )
    print(f"\nWrote {out_j}")
    print(f"Wrote {out_c}")


if __name__ == "__main__":
    main()
