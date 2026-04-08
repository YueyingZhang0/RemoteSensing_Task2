#!/usr/bin/env python3
"""Build fine-tune multi-seed tables (FT-1 CHASE, FT-2 HRF, FT-3 zero-shot → fine-tune deltas).

Reads **test-set** external metrics from fine-tune dirs using prefix ``external_test_`` (same as
``build_external_adaptation_tables.py``). Zero-shot rows use no prefix.

Missing run directories (e.g. seed40 not run yet) are skipped; summary mean±std uses only available seeds.

Outputs:
  - tables/ft_multiseed_FT1_chase.{md,csv}
  - tables/ft_multiseed_FT2_hrf.{md,csv}
  - tables/ft_multiseed_FT3_delta.{md,csv}
  - paper_assets/ft_multiseed_tables.json
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

FT_PREFIX = "external_test_"

NOTE_TEST = (
    "Metrics are **test split** means (eval_external_thinvessel protocol @0.5), "
    "stored as external_test_* in fine-tune run dirs."
)


def _read_json(p: Path) -> Dict[str, Any]:
    if not p.is_file():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))


def _load_ft_metrics(run_dir: Path) -> Optional[Dict[str, Any]]:
    val_p = run_dir / f"{FT_PREFIX}val_metrics.json"
    if not val_p.is_file():
        return None
    val = _read_json(val_p)
    hard = _read_json(run_dir / f"{FT_PREFIX}hard_metrics_thinvessel.json")
    cld = _read_json(run_dir / f"{FT_PREFIX}cldice_summary.json")
    return {
        "dice": float(val["val_dice_mean"]),
        "recall": float(val["val_recall_mean"]),
        "cldice": float(cld["cldice_mean"]),
        "hard_dice": float(hard["hard_dice_mean"]),
    }


def _load_zs_metrics(run_dir: Path) -> Dict[str, Any]:
    val = _read_json(run_dir / "val_metrics.json")
    hard = _read_json(run_dir / "hard_metrics_thinvessel.json")
    cld = _read_json(run_dir / "cldice_summary.json")
    return {
        "dice": float(val["val_dice_mean"]),
        "recall": float(val["val_recall_mean"]),
        "cldice": float(cld["cldice_mean"]),
        "hard_dice": float(hard["hard_dice_mean"]),
    }


def _fmt(x: Any) -> str:
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def _write_md(path: Path, cols: List[str], rows: List[Dict[str, Any]], title: str, extra: str = "") -> None:
    lines = [f"## {title}", ""]
    if extra:
        lines.append(extra)
        lines.append("")
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r[c]) for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, cols: List[str], rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def _mean_std(vals: List[float]) -> Tuple[float, float]:
    a = np.array(vals, dtype=np.float64)
    return float(np.mean(a)), float(np.std(a, ddof=0))


def _ft1_rows(
    *,
    dataset_label: str,
    name_prefix: str,
    seeds: List[int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (detail_rows, summary_rows). name_prefix: 'ft_chase' or 'ft_hrf'."""
    method_map = [("P0", "p0"), ("CDC (J3)", "j3")]
    detail: List[Dict[str, Any]] = []
    by_method: Dict[str, List[Dict[str, float]]] = {"P0": [], "CDC (J3)": []}

    for method_label, short in method_map:
        for seed in seeds:
            run_dir = ROOT / "outputs" / f"{name_prefix}_{short}_seed{seed}"
            m = _load_ft_metrics(run_dir)
            if m is None:
                continue
            detail.append(
                {
                    "Method": method_label,
                    "seed": seed,
                    "Dice @0.5": m["dice"],
                    "Recall @0.5": m["recall"],
                    "clDice @0.5": m["cldice"],
                    "Hard Dice @0.5 (thin-vessel proxy)": m["hard_dice"],
                    "run_dir": str(run_dir.relative_to(ROOT)),
                }
            )
            by_method[method_label].append(m)

    summary: List[Dict[str, Any]] = []
    for method_label in ("P0", "CDC (J3)"):
        rows_m = by_method[method_label]
        if not rows_m:
            continue
        dices = [x["dice"] for x in rows_m]
        recs = [x["recall"] for x in rows_m]
        clds = [x["cldice"] for x in rows_m]
        hards = [x["hard_dice"] for x in rows_m]
        dm, ds = _mean_std(dices)
        rm, rs = _mean_std(recs)
        cm, cs = _mean_std(clds)
        hm, hs = _mean_std(hards)
        summary.append(
            {
                "Dataset": dataset_label,
                "Method": method_label,
                "n_seeds": len(rows_m),
                "Dice mean": dm,
                "Dice std": ds,
                "Recall mean": rm,
                "Recall std": rs,
                "clDice mean": cm,
                "clDice std": cs,
                "Hard Dice mean": hm,
                "Hard Dice std": hs,
            }
        )

    return detail, summary


def _ft3_deltas(
    *,
    dataset_label: str,
    zs_p0_dir: Path,
    zs_j3_dir: Path,
    name_prefix: str,
    seeds: List[int],
) -> List[Dict[str, Any]]:
    zs_p0 = _load_zs_metrics(zs_p0_dir)
    zs_j3 = _load_zs_metrics(zs_j3_dir)
    out: List[Dict[str, Any]] = []

    for method_label, short, zs in (
        ("P0", "p0", zs_p0),
        ("CDC (J3)", "j3", zs_j3),
    ):
        for seed in seeds:
            ft_dir = ROOT / "outputs" / f"{name_prefix}_{short}_seed{seed}"
            ft = _load_ft_metrics(ft_dir)
            if ft is None:
                continue
            out.append(
                {
                    "Dataset": dataset_label,
                    "Method": method_label,
                    "seed": seed,
                    "ΔDice": ft["dice"] - zs["dice"],
                    "ΔRecall": ft["recall"] - zs["recall"],
                    "ΔclDice": ft["cldice"] - zs["cldice"],
                    "ΔHard Dice (thin-vessel proxy)": ft["hard_dice"] - zs["hard_dice"],
                    "run_dir": str(ft_dir.relative_to(ROOT)),
                }
            )
    return out


def main() -> None:
    seeds = [42, 40]
    tables = ROOT / "tables"
    pa = ROOT / "paper_assets"
    tables.mkdir(parents=True, exist_ok=True)
    pa.mkdir(parents=True, exist_ok=True)

    cols_ft = ["Method", "seed", "Dice @0.5", "Recall @0.5", "clDice @0.5", "Hard Dice @0.5 (thin-vessel proxy)", "run_dir"]
    cols_sum = [
        "Dataset",
        "Method",
        "n_seeds",
        "Dice mean",
        "Dice std",
        "Recall mean",
        "Recall std",
        "clDice mean",
        "clDice std",
        "Hard Dice mean",
        "Hard Dice std",
    ]
    cols_delta = [
        "Dataset",
        "Method",
        "seed",
        "ΔDice",
        "ΔRecall",
        "ΔclDice",
        "ΔHard Dice (thin-vessel proxy)",
        "run_dir",
    ]

    chase_detail, chase_sum = _ft1_rows(
        dataset_label="CHASE_DB1 (1st observer)",
        name_prefix="ft_chase",
        seeds=seeds,
    )
    hrf_detail, hrf_sum = _ft1_rows(
        dataset_label="HRF all (manual1; FOV mask)",
        name_prefix="ft_hrf",
        seeds=seeds,
    )

    md_note = NOTE_TEST + " Mean±std over available seeds only (typically n=2); interpret cautiously."

    _write_md(
        tables / "ft_multiseed_FT1_chase.md",
        cols_ft,
        chase_detail,
        "FT-1. CHASE_DB1 fine-tune (test metrics) by seed",
        md_note,
    )
    _write_csv(tables / "ft_multiseed_FT1_chase.csv", cols_ft, chase_detail)

    _write_md(
        tables / "ft_multiseed_FT1_chase_summary.md",
        cols_sum,
        chase_sum,
        "FT-1 summary. CHASE mean ± std by method",
        md_note,
    )
    _write_csv(tables / "ft_multiseed_FT1_chase_summary.csv", cols_sum, chase_sum)

    _write_md(
        tables / "ft_multiseed_FT2_hrf.md",
        cols_ft,
        hrf_detail,
        "FT-2. HRF fine-tune (test metrics) by seed",
        md_note,
    )
    _write_csv(tables / "ft_multiseed_FT2_hrf.csv", cols_ft, hrf_detail)

    _write_md(
        tables / "ft_multiseed_FT2_hrf_summary.md",
        cols_sum,
        hrf_sum,
        "FT-2 summary. HRF mean ± std by method",
        md_note,
    )
    _write_csv(tables / "ft_multiseed_FT2_hrf_summary.csv", cols_sum, hrf_sum)

    delta_rows: List[Dict[str, Any]] = []
    delta_rows.extend(
        _ft3_deltas(
            dataset_label="CHASE_DB1 (1st observer)",
            zs_p0_dir=ROOT / "outputs" / "ext_chase_p0_zeroshot",
            zs_j3_dir=ROOT / "outputs" / "ext_chase_j3_zeroshot",
            name_prefix="ft_chase",
            seeds=seeds,
        )
    )
    delta_rows.extend(
        _ft3_deltas(
            dataset_label="HRF all (manual1; FOV mask)",
            zs_p0_dir=ROOT / "outputs" / "ext_hrf_p0_zeroshot",
            zs_j3_dir=ROOT / "outputs" / "ext_hrf_j3_zeroshot",
            name_prefix="ft_hrf",
            seeds=seeds,
        )
    )

    _write_md(
        tables / "ft_multiseed_FT3_delta.md",
        cols_delta,
        delta_rows,
        "FT-3. Fine-tune minus zero-shot (test split, same protocol)",
        NOTE_TEST,
    )
    _write_csv(tables / "ft_multiseed_FT3_delta.csv", cols_delta, delta_rows)

    payload = {
        "note": NOTE_TEST,
        "seeds_requested": seeds,
        "FT1_chase": chase_detail,
        "FT1_chase_summary": chase_sum,
        "FT2_hrf": hrf_detail,
        "FT2_hrf_summary": hrf_sum,
        "FT3_delta": delta_rows,
    }
    (pa / "ft_multiseed_tables.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"FT1 CHASE rows: {len(chase_detail)}; FT2 HRF rows: {len(hrf_detail)}; FT3 rows: {len(delta_rows)}")
    print(f"Wrote {tables / 'ft_multiseed_FT1_chase.md'} etc.")


if __name__ == "__main__":
    main()
