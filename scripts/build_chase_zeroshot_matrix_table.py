#!/usr/bin/env python3
"""Build CHASE_DB1 zero-shot comparison Table X (fixed primary metric: Hard Dice @0.5 thin-vessel proxy).

Reads ``val_metrics.json`` / ``hard_metrics_thinvessel.json`` / ``cldice_summary.json`` from each run dir.
Expects **new** output directories (do not overwrite canonical ``outputs/task3_*``).

Default run dirs (create by running eval scripts):
  - outputs/ext_chase_p0_zeroshot
  - outputs/ext_chase_j3_zeroshot
  - outputs/ext_chase_attention_unet_zeroshot
  - outputs/ext_chase_swin_unet_zeroshot
  - outputs/ext_chase_nnunet_v2_zeroshot

Outputs:
  - tables/chase_zeroshot_matrix_tableX.md
  - tables/chase_zeroshot_matrix_tableX.csv
  - paper_assets/chase_zeroshot_matrix.json

Fill ``brief_conclusions`` in the JSON after you inspect numbers (script writes a template only).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]

PRIMARY_NOTE = (
    "Primary metric for this matrix: **Hard Dice @0.5** (thin-vessel proxy mean). "
    "Dice / Recall / clDice @0.5 are secondary and were fixed a priori."
)
NNUNET_NOTE = (
    "nnU-Net v2 row: standard nnU-Net preprocessing / resampling / normalization at inference; "
    "probabilities resized to CHASE GT size before @0.5 (not the raw sliding-window PyTorch protocol)."
)


def _read_json(p: Path) -> Dict[str, Any]:
    if not p.is_file():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))


HK = "Hard Dice @0.5 (thin-vessel proxy)"
DK = "Dice @0.5"
RK = "Recall @0.5"
CK = "clDice @0.5"


def _row_by_id(rows: List[Dict[str, Any]], internal_id: str) -> Optional[Dict[str, Any]]:
    for r in rows:
        if r.get("internal_id") == internal_id:
            return r
    return None


def _brief_conclusions(rows: List[Dict[str, Any]]) -> Dict[str, str]:
    """Fixed four answers from table numbers (primary = Hard Dice). All rows must be present."""
    p0 = _row_by_id(rows, "p0_baseline_zeroshot_chase")
    j3 = _row_by_id(rows, "cdc_j3_zeroshot_chase")
    att = _row_by_id(rows, "attention_unet_baseline_zeroshot_chase")
    nn = _row_by_id(rows, "nnunet_v2_zeroshot_chase")
    if not all([p0, j3, att, nn]):
        return {
            "1_cdc_vs_p0": "(incomplete matrix rows)",
            "2_cdc_vs_nnunet": "(incomplete matrix rows)",
            "3_cdc_vs_attention": "(incomplete matrix rows)",
            "4_best_overall_connectivity_hard": "(incomplete matrix rows)",
        }

    def fmt4(x: float) -> str:
        return f"{float(x):.4f}"

    j3_stronger_p0 = (
        float(j3[HK]) > float(p0[HK])
        and float(j3[DK]) > float(p0[DK])
        and float(j3[CK]) > float(p0[CK])
        and float(j3[RK]) > float(p0[RK])
    )
    j3_stronger_nn = (
        float(j3[HK]) > float(nn[HK])
        and float(j3[DK]) > float(nn[DK])
        and float(j3[CK]) > float(nn[CK])
        and float(j3[RK]) > float(nn[RK])
    )
    j3_stronger_att = (
        float(j3[HK]) > float(att[HK])
        and float(j3[DK]) > float(att[DK])
        and float(j3[CK]) > float(att[CK])
        and float(j3[RK]) > float(att[RK])
    )

    def best_for(key: str) -> str:
        best = max(rows, key=lambda r: float(r[key]))
        return str(best["Method"])

    best_d = best_for(DK)
    best_c = best_for(CK)
    best_h = best_for(HK)
    same_all = best_d == best_c == best_h

    c1 = (
        f"CDC/J3 相对 P0 {'更强' if j3_stronger_p0 else '并非在全部指标上更强'}："
        f"Hard Dice {fmt4(j3[HK])} vs {fmt4(p0[HK])}；Dice {fmt4(j3[DK])} vs {fmt4(p0[DK])}；"
        f"clDice {fmt4(j3[CK])} vs {fmt4(p0[CK])}；Recall {fmt4(j3[RK])} vs {fmt4(p0[RK])}。"
    )
    c2 = (
        f"CDC/J3 相对 nnU-Net v2 {'更强' if j3_stronger_nn else '不更强（nnU-Net 在 Hard Dice / Dice / clDice 上更高）'}"
        f"；nnU-Net 使用标准预处理协议，见 table_notes.nnunet。"
    )
    c3 = (
        f"CDC/J3 相对 Attention U-Net {'更强' if j3_stronger_att else '不更强（Attention U-Net 在 Hard Dice / Dice / clDice 上更高）'}"
        f"；二者均为 raw-image 滑窗协议。"
    )
    if same_all:
        c4 = f"Dice、clDice、Hard Dice（thin proxy）三项最高均为：{best_d}。"
    else:
        c4 = (
            f"分项最优：整体重叠（Dice）— {best_d}；连通性（clDice）— {best_c}；"
            f"细血管硬区（Hard Dice）— {best_h}。"
        )
    return {
        "1_cdc_vs_p0": c1,
        "2_cdc_vs_nnunet": c2,
        "3_cdc_vs_attention": c3,
        "4_best_overall_connectivity_hard": c4,
    }


def _load_row(run_dir: Path, *, paper_method: str, internal_id: str) -> Dict[str, Any]:
    val = _read_json(run_dir / "val_metrics.json")
    hard = _read_json(run_dir / "hard_metrics_thinvessel.json")
    cld = _read_json(run_dir / "cldice_summary.json")
    proto = str(val.get("inference_protocol", "raw_image_sliding_window"))
    return {
        "Method": paper_method,
        "Hard Dice @0.5 (thin-vessel proxy)": float(hard["hard_dice_mean"]),
        "Dice @0.5": float(val["val_dice_mean"]),
        "Recall @0.5": float(val["val_recall_mean"]),
        "clDice @0.5": float(cld["cldice_mean"]),
        "internal_id": internal_id,
        "inference_protocol": proto,
    }


def main() -> None:
    rows_spec: List[Dict[str, Any]] = [
        {
            "run_dir": ROOT / "outputs" / "ext_chase_p0_zeroshot",
            "paper_method": "P0 U-Net baseline (zero-shot)",
            "internal_id": "p0_baseline_zeroshot_chase",
        },
        {
            "run_dir": ROOT / "outputs" / "ext_chase_j3_zeroshot",
            "paper_method": "CDC / J3 joint (zero-shot)",
            "internal_id": "cdc_j3_zeroshot_chase",
        },
        {
            "run_dir": ROOT / "outputs" / "ext_chase_attention_unet_zeroshot",
            "paper_method": "Attention U-Net baseline (zero-shot)",
            "internal_id": "attention_unet_baseline_zeroshot_chase",
        },
        {
            "run_dir": ROOT / "outputs" / "ext_chase_swin_unet_zeroshot",
            "paper_method": "Swin-UNet baseline (zero-shot)",
            "internal_id": "swin_unet_baseline_zeroshot_chase",
        },
        {
            "run_dir": ROOT / "outputs" / "ext_chase_nnunet_v2_zeroshot",
            "paper_method": "nnU-Net v2 (zero-shot)",
            "internal_id": "nnunet_v2_zeroshot_chase",
        },
    ]

    table_rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    for spec in rows_spec:
        rd = spec["run_dir"]
        if not (rd / "hard_metrics_thinvessel.json").is_file():
            missing.append(str(rd.relative_to(ROOT)))
            continue
        r = _load_row(rd, paper_method=spec["paper_method"], internal_id=spec["internal_id"])
        r["run_dir"] = str(rd.relative_to(ROOT))
        table_rows.append(r)

    cols = [
        "Method",
        "Hard Dice @0.5 (thin-vessel proxy)",
        "Dice @0.5",
        "Recall @0.5",
        "clDice @0.5",
        "internal_id",
    ]

    md_path = ROOT / "tables" / "chase_zeroshot_matrix_tableX.md"
    csv_path = ROOT / "tables" / "chase_zeroshot_matrix_tableX.csv"
    json_path = ROOT / "paper_assets" / "chase_zeroshot_matrix.json"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "## Table X — CHASE_DB1 zero-shot (thin-vessel primary)",
        "",
        PRIMARY_NOTE,
        "",
        NNUNET_NOTE,
        "",
    ]
    if missing:
        lines.append("### Missing run dirs (skipped rows)")
        lines.extend(f"- `{m}`" for m in missing)
        lines.append("")

    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for r in table_rows:
        lines.append("| " + " | ".join(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in table_rows:
            w.writerow({c: r[c] for c in cols})

    payload = {
        "primary_metric": "Hard Dice @0.5 (thin-vessel proxy mean)",
        "secondary_metrics": ["Dice @0.5", "Recall @0.5", "clDice @0.5"],
        "table_notes": {"protocol": PRIMARY_NOTE, "nnunet": NNUNET_NOTE},
        "split_reference": "outputs/ext_chase_split_seed42/test_split_ids.json (CHASE_DB1 1stHO; 16 images unless split file changed).",
        "rows": table_rows,
        "missing_run_dirs": missing,
        "brief_conclusions": _brief_conclusions(table_rows) if not missing and len(table_rows) >= 5 else {},
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote {md_path} ({len(table_rows)} rows); missing={len(missing)}")


if __name__ == "__main__":
    main()
