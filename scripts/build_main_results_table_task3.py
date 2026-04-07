#!/usr/bin/env python3
"""Build the Task-3 main paper table from existing run summaries (no training / no inference).

Outputs:
  - tables/main_results_task3.md
  - tables/main_results_task3.csv
  - paper_assets/main_results_task3.json

The table follows the paper's unified reporting convention:
  - fixed threshold = 0.5 for main metrics (Global Dice, Hard Dice, clDice Hard)
  - best-threshold metrics are supportive only (Primary Hard Dice, best threshold τ)
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class MethodSpec:
    internal_id: str
    paper_label: str
    run_dir: Optional[Path] = None
    cldice_key: Optional[str] = None
    nnunet_cldice_path: Optional[Path] = None


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _tau_key(tau: float) -> str:
    # Matches existing threshold_metrics.json keys, e.g. "0.50", "0.30".
    return f"{float(tau):.2f}"


def _get_required(d: Dict[str, Any], path: str) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"Missing key path '{path}' (stopped at '{part}')")
        cur = cur[part]
    return cur


def _load_cldice_hard_05(
    *,
    cldice_summary: Dict[str, Any],
    spec: MethodSpec,
) -> float:
    if spec.cldice_key is None:
        raise ValueError(f"{spec.internal_id}: cldice_key is required for cldice_summary lookup")
    methods = _get_required(cldice_summary, "methods")
    if spec.cldice_key not in methods:
        raise KeyError(f"cldice_summary.json missing methods[{spec.cldice_key!r}] for {spec.internal_id}")
    # Note: the key contains a dot ("cldice_hard_0.5"), so index directly.
    v = methods[spec.cldice_key]["cldice_hard_0.5"]
    return float(v)


def _load_nnunet_cldice_hard_05(*, nnunet_cldice: Dict[str, Any]) -> float:
    v = _get_required(nnunet_cldice, "cldice.cldice_hard")
    return float(v)


def _load_threshold_metrics(run_dir: Path) -> Dict[str, Any]:
    return _read_json(run_dir / "threshold_metrics.json")


def _extract_row(
    *,
    spec: MethodSpec,
    cldice_summary: Dict[str, Any],
    nnunet_cldice: Dict[str, Any],
    fixed_tau: float = 0.5,
) -> Dict[str, Any]:
    if spec.run_dir is None:
        raise ValueError(f"{spec.internal_id}: run_dir is required")
    tm = _load_threshold_metrics(spec.run_dir)

    fixed_key = _tau_key(fixed_tau)
    best_tau = float(_get_required(tm, "best_hard_dice_threshold"))
    best_key = _tau_key(best_tau)

    per = _get_required(tm, "per_threshold")
    if fixed_key not in per:
        raise KeyError(f"{spec.internal_id}: per_threshold missing fixed key {fixed_key!r} in {spec.run_dir}")
    if best_key not in per:
        raise KeyError(
            f"{spec.internal_id}: per_threshold missing best key {best_key!r} (from best_hard_dice_threshold={best_tau})"
        )

    # Note: per_threshold keys are strings like "0.50" which contain a dot,
    # so we must index them directly instead of using the dot-path helper.
    global_dice_05 = float(per[fixed_key]["val_dice_mean"])
    hard_dice_05 = float(per[fixed_key]["hard_dice_mean"])
    primary_hard_dice = float(per[best_key]["hard_dice_mean"])

    if spec.internal_id == "nnunet_v2":
        cldice_hard_05 = _load_nnunet_cldice_hard_05(nnunet_cldice=nnunet_cldice)
    else:
        cldice_hard_05 = _load_cldice_hard_05(cldice_summary=cldice_summary, spec=spec)

    return {
        "Method (paper)": spec.paper_label,
        "internal_id": spec.internal_id,
        "Global Dice @0.5": global_dice_05,
        "Hard Dice @0.5": hard_dice_05,
        "Primary Hard Dice": primary_hard_dice,
        "best τ": best_tau,
        "clDice Hard @0.5": cldice_hard_05,
        "source_threshold_metrics": str((spec.run_dir / "threshold_metrics.json").resolve()),
        "source_cldice": str(
            (spec.nnunet_cldice_path.resolve() if spec.internal_id == "nnunet_v2" else (ROOT / "cldice_summary.json").resolve())
        ),
    }


def main() -> None:
    fixed_tau = 0.5

    # Single manifest: this is the only place that maps methods -> paths.
    methods: List[MethodSpec] = [
        MethodSpec(
            internal_id="P0_baseline",
            paper_label="Baseline (P0)",
            run_dir=ROOT / "outputs" / "task3_v2_1_baseline_same_split",
            cldice_key="P0_baseline",
        ),
        MethodSpec(
            internal_id="P5_oracle_matched_trigger",
            paper_label="Oracle-weighted reference (P5)",
            run_dir=ROOT / "outputs" / "task3_v3_1_oracle_matched_trigger_same_split",
            cldice_key="P5_oracle_matched_trigger",
        ),
        MethodSpec(
            internal_id="J3_joint_detach_false",
            paper_label="CDC (proposed; internal J3)",
            run_dir=ROOT / "outputs" / "task3_J3_joint_detach_false",
            cldice_key="J3_joint_detach_false",
        ),
        MethodSpec(
            internal_id="attention_unet_baseline",
            paper_label="Attention U-Net baseline",
            run_dir=ROOT / "outputs" / "task3_attention_unet_baseline",
            cldice_key="attention_unet_baseline",
        ),
        MethodSpec(
            internal_id="swin_unet_baseline",
            paper_label="Swin-UNet baseline",
            run_dir=ROOT / "outputs" / "task3_swin_unet_baseline",
            cldice_key="swin_unet_baseline",
        ),
        MethodSpec(
            internal_id="nnunet_v2",
            paper_label="nnU-Net v2 baseline",
            run_dir=ROOT / "outputs" / "task3_nnunet_eval",
            nnunet_cldice_path=ROOT / "outputs" / "task3_nnunet_eval" / "nnunet_cldice.json",
        ),
    ]

    cldice_summary_path = ROOT / "cldice_summary.json"
    cldice_summary = _read_json(cldice_summary_path)
    if float(_get_required(cldice_summary, "threshold_fixed")) != fixed_tau:
        raise ValueError(
            f"cldice_summary.json threshold_fixed mismatch: expected {fixed_tau}, got {cldice_summary.get('threshold_fixed')}"
        )

    nnunet_cldice_path = ROOT / "outputs" / "task3_nnunet_eval" / "nnunet_cldice.json"
    nnunet_cldice = _read_json(nnunet_cldice_path)

    rows: List[Dict[str, Any]] = []
    for spec in methods:
        rows.append(
            _extract_row(
                spec=spec,
                cldice_summary=cldice_summary,
                nnunet_cldice=nnunet_cldice,
                fixed_tau=fixed_tau,
            )
        )

    # Write CSV/MD (human-facing) + JSON (machine-facing, with provenance).
    tables_dir = ROOT / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tables_dir / "main_results_task3.csv"
    md_path = tables_dir / "main_results_task3.md"

    cols = [
        "Method (paper)",
        "Global Dice @0.5",
        "Hard Dice @0.5",
        "clDice Hard @0.5",
        "Primary Hard Dice",
        "best τ",
        "internal_id",
    ]

    def fmt(x: Any) -> str:
        if isinstance(x, float):
            return f"{x:.4f}"
        return str(x)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in cols})

    md_lines: List[str] = []
    md_lines.append("## Task 3 main results (canonical patch / group-split)")
    md_lines.append("")
    md_lines.append(f"Unified reporting convention: fixed threshold \(\\tau={fixed_tau}\\) for main endpoints.")
    md_lines.append("")
    md_lines.append("| " + " | ".join(cols) + " |")
    md_lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for r in rows:
        md_lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    paper_assets_dir = ROOT / "paper_assets"
    paper_assets_dir.mkdir(parents=True, exist_ok=True)
    json_path = paper_assets_dir / "main_results_task3.json"
    payload = {
        "protocol": "task3_patch_group_split_canonical",
        "evaluation_type": "canonical_patch_group_split",
        "fixed_threshold": fixed_tau,
        "columns": cols,
        "methods_manifest": [
            {
                "internal_id": m.internal_id,
                "paper_label": m.paper_label,
                "run_dir": (str(m.run_dir.resolve()) if m.run_dir else None),
                "cldice_key": m.cldice_key,
                "nnunet_cldice_path": (str(m.nnunet_cldice_path.resolve()) if m.nnunet_cldice_path else None),
            }
            for m in methods
        ],
        "rows": rows,
        "source_files": sorted(
            {
                str(cldice_summary_path.resolve()),
                str(nnunet_cldice_path.resolve()),
                *[str((m.run_dir / "threshold_metrics.json").resolve()) for m in methods if m.run_dir],
            }
        ),
        "notes": {
            "main_endpoints": [
                "Global Dice @0.5",
                "Hard Dice @0.5",
                "clDice Hard @0.5 (supportive topology metric, not primary endpoint)",
            ],
            "supporting_endpoints": [
                "Primary Hard Dice (at best_hard_dice_threshold)",
                "best τ (best_hard_dice_threshold)",
            ],
        },
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote {md_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()

