#!/usr/bin/env python3
"""Build external adaptation tables (thin-vessel proxy hard Dice) without touching canonical main table.

Outputs:
  - tables/external_thinvessel_tableA.{md,csv}
  - tables/external_thinvessel_tableB_delta.{md,csv}
  - paper_assets/external_thinvessel_tables.json
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _read_json(p: Path) -> Dict[str, Any]:
    if not p.is_file():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))


def _load_metrics(run_dir: Path, *, prefix: str = "") -> Dict[str, Any]:
    val = _read_json(run_dir / f"{prefix}val_metrics.json")
    hard = _read_json(run_dir / f"{prefix}hard_metrics_thinvessel.json")
    cld = _read_json(run_dir / f"{prefix}cldice_summary.json")
    return {
        "dice": float(val["val_dice_mean"]),
        "recall": float(val["val_recall_mean"]),
        "cldice": float(val["cldice_mean"]),
        "hard_dice": float(hard["hard_dice_mean"]),
        "dataset": str(val.get("dataset", "")),
        "threshold": float(val.get("threshold", 0.5)),
        "thin_cfg": hard.get("thin_vessel_proxy", {}),
    }


def _fmt(x: Any) -> str:
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def _write_md(path: Path, cols: List[str], rows: List[Dict[str, Any]], title: str) -> None:
    lines = [f"## {title}", ""]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r[c]) for c in cols) + " |")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_csv(path: Path, cols: List[str], rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def main() -> None:
    # Canonical checkpoints re-evaluated in external-style thin-vessel protocol
    drive_p0 = ROOT / "outputs" / "ext_drive_p0_thinvessel_eval"
    drive_j3 = ROOT / "outputs" / "ext_drive_j3_thinvessel_eval"
    chase_zs_p0 = ROOT / "outputs" / "ext_chase_p0_zeroshot"
    chase_zs_j3 = ROOT / "outputs" / "ext_chase_j3_zeroshot"
    chase_ft_p0 = ROOT / "outputs" / "ft_chase_p0_seed42"
    chase_ft_j3 = ROOT / "outputs" / "ft_chase_j3_seed42"
    hrf_zs_p0 = ROOT / "outputs" / "ext_hrf_p0_zeroshot"
    hrf_zs_j3 = ROOT / "outputs" / "ext_hrf_j3_zeroshot"
    hrf_ft_p0 = ROOT / "outputs" / "ft_hrf_p0_seed42"
    hrf_ft_j3 = ROOT / "outputs" / "ft_hrf_j3_seed42"

    # Fine-tune dirs store external-style test metrics with prefix to avoid overwriting training val_metrics.json.
    ft_prefix = "external_test_"

    tableA: List[Dict[str, Any]] = []

    def add(dataset: str, method: str, setting: str, metrics: Dict[str, Any], n_train: int, n_val: int, n_test: int, seed: str, run_dir: Path) -> None:
        tableA.append(
            {
                "Dataset": dataset,
                "Method": method,
                "Setting": setting,
                "Dice @0.5": metrics["dice"],
                "Recall @0.5": metrics["recall"],
                "clDice @0.5": metrics["cldice"],
                "Hard Dice @0.5 (thin-vessel proxy)": metrics["hard_dice"],
                "n_train": n_train,
                "n_val": n_val,
                "n_test": n_test,
                "seed": seed,
                "run_dir": str(run_dir.relative_to(ROOT)),
            }
        )

    add("DRIVE (training subset only in repo)", "P0", "re-eval", _load_metrics(drive_p0), 0, 0, 20, "N/A", drive_p0)
    add("DRIVE (training subset only in repo)", "CDC (J3)", "re-eval", _load_metrics(drive_j3), 0, 0, 20, "N/A", drive_j3)

    add("CHASE_DB1 (1st observer)", "P0", "zero-shot", _load_metrics(chase_zs_p0), 8, 4, 16, "42", chase_zs_p0)
    add("CHASE_DB1 (1st observer)", "CDC (J3)", "zero-shot", _load_metrics(chase_zs_j3), 8, 4, 16, "42", chase_zs_j3)
    add("CHASE_DB1 (1st observer)", "P0", "fine-tuned", _load_metrics(chase_ft_p0, prefix=ft_prefix), 8, 4, 16, "42", chase_ft_p0)
    add("CHASE_DB1 (1st observer)", "CDC (J3)", "fine-tuned", _load_metrics(chase_ft_j3, prefix=ft_prefix), 8, 4, 16, "42", chase_ft_j3)

    add("HRF all (manual1; FOV mask)", "P0", "zero-shot", _load_metrics(hrf_zs_p0), 10, 5, 30, "42", hrf_zs_p0)
    add("HRF all (manual1; FOV mask)", "CDC (J3)", "zero-shot", _load_metrics(hrf_zs_j3), 10, 5, 30, "42", hrf_zs_j3)
    add("HRF all (manual1; FOV mask)", "P0", "fine-tuned", _load_metrics(hrf_ft_p0, prefix=ft_prefix), 10, 5, 30, "42", hrf_ft_p0)
    add("HRF all (manual1; FOV mask)", "CDC (J3)", "fine-tuned", _load_metrics(hrf_ft_j3, prefix=ft_prefix), 10, 5, 30, "42", hrf_ft_j3)

    colsA = [
        "Dataset",
        "Method",
        "Setting",
        "Dice @0.5",
        "Recall @0.5",
        "clDice @0.5",
        "Hard Dice @0.5 (thin-vessel proxy)",
        "n_train",
        "n_val",
        "n_test",
        "seed",
    ]

    tables_dir = ROOT / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    mdA = tables_dir / "external_thinvessel_tableA.md"
    csvA = tables_dir / "external_thinvessel_tableA.csv"
    _write_md(mdA, colsA, tableA, "Table A. External-style evaluation (dataset-agnostic thin-vessel hard proxy)")
    _write_csv(csvA, colsA, tableA)

    # Table B: deltas (fine-tuned - zero-shot) per external dataset
    def by(dataset: str, method: str, setting: str) -> Dict[str, Any]:
        for r in tableA:
            if r["Dataset"] == dataset and r["Method"] == method and r["Setting"] == setting:
                return r
        raise KeyError((dataset, method, setting))

    deltas: List[Dict[str, Any]] = []
    for ds_label in ("CHASE_DB1 (1st observer)", "HRF all (manual1; FOV mask)"):
        for method in ["P0", "CDC (J3)"]:
            zs = by(ds_label, method, "zero-shot")
            ft = by(ds_label, method, "fine-tuned")
            deltas.append(
                {
                    "Dataset": ds_label,
                    "Method": method,
                    "ΔDice": float(ft["Dice @0.5"] - zs["Dice @0.5"]),
                    "ΔRecall": float(ft["Recall @0.5"] - zs["Recall @0.5"]),
                    "ΔclDice": float(ft["clDice @0.5"] - zs["clDice @0.5"]),
                    "ΔHard Dice (thin-vessel proxy)": float(
                        ft["Hard Dice @0.5 (thin-vessel proxy)"] - zs["Hard Dice @0.5 (thin-vessel proxy)"]
                    ),
                    "seed": ft["seed"],
                }
            )

    colsB = ["Dataset", "Method", "ΔDice", "ΔRecall", "ΔclDice", "ΔHard Dice (thin-vessel proxy)", "seed"]
    mdB = tables_dir / "external_thinvessel_tableB_delta.md"
    csvB = tables_dir / "external_thinvessel_tableB_delta.csv"
    _write_md(mdB, colsB, deltas, "Table B. Fine-tune deltas (fine-tuned − zero-shot)")
    _write_csv(csvB, colsB, deltas)

    pa = ROOT / "paper_assets"
    pa.mkdir(parents=True, exist_ok=True)
    payload = {
        "tableA": tableA,
        "tableB_delta": deltas,
        "source_runs": [
            str(x)
            for x in [
                drive_p0,
                drive_j3,
                chase_zs_p0,
                chase_zs_j3,
                chase_ft_p0,
                chase_ft_j3,
                hrf_zs_p0,
                hrf_zs_j3,
                hrf_ft_p0,
                hrf_ft_j3,
            ]
        ],
    }
    (pa / "external_thinvessel_tables.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote {mdA}")
    print(f"Wrote {mdB}")


if __name__ == "__main__":
    main()

