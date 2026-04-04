from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.io import load_json, save_json, save_txt


def _val_loss_improved_from_history(val_losses: list) -> tuple[bool | None, str]:
    """Return (improved, note). None means not computable (missing or too short)."""
    if not val_losses:
        return None, "no val_loss in history"
    losses = [float(x) for x in val_losses]
    n = len(losses)
    if n < 2:
        return None, "fewer than 2 epochs in history"
    third = max(1, n // 3)
    early = sum(losses[:third]) / third
    late = sum(losses[-third:]) / third
    improved = late < early - 1e-9
    return improved, f"mean early ({third} ep) vs late ({third} ep): {early:.6f} -> {late:.6f}"


def save_top_bottom_predictions(
    names: List[str],
    preds: np.ndarray,
    df_meta: pd.DataFrame,
    save_path: str | Path,
    top_k: int = 20,
    descending: bool = True,
) -> None:
    order = np.argsort(-preds if descending else preds)
    lines = []
    for i in order[:top_k]:
        name = names[int(i)]
        p = float(preds[int(i)])
        sub = df_meta[df_meta["patch_name"] == name]
        if len(sub) == 0:
            lines.append(f"{name}\tpred={p:.6f}\t(no metadata)\n")
            continue
        r = sub.iloc[0]
        gt = float(r.get("sci_res2_norm", float("nan")))
        lines.append(
            f"{name}\tpred={p:.6f}\tgt={gt:.6f}\t"
            f"vessel_area={int(r['vessel_area'])}\tfov_ratio={float(r['fov_ratio']):.4f}\t"
            f"junctions={int(r['junction_count'])}\tend_eff={int(r['endpoint_eff'])}\n"
        )
    save_txt(lines, save_path)


def generate_analysis_report(
    cnn_metrics: dict,
    baseline_json_path: str | Path,
    output_dir: str | Path,
    pearson_min: float = 0.35,
    spearman_min: float = 0.40,
    pearson_margin: float = 0.10,
    manual_visual_ok: bool | None = None,
) -> dict:
    out = Path(output_dir)

    scalar_pearsons: Dict[str, float] = {}
    baselines: Optional[dict] = None
    bp = Path(baseline_json_path)
    if bp.is_file():
        baselines = load_json(bp)
        for name, item in baselines.items():
            if isinstance(item, dict) and "val" in item and "pearson" in item["val"]:
                scalar_pearsons[name] = float(item["val"]["pearson"])

    best_scalar = max(scalar_pearsons.values()) if scalar_pearsons else float("-inf")

    pass_pearson = cnn_metrics["pearson"] > pearson_min
    pass_spearman = cnn_metrics["spearman"] > spearman_min
    pass_margin = cnn_metrics["pearson"] > best_scalar + pearson_margin if scalar_pearsons else True

    val_loss_improved: bool | None = None
    val_loss_improved_note = ""
    hist_path = out / "history.json"
    if hist_path.is_file():
        hist = load_json(hist_path)
        vl = hist.get("val_loss") if isinstance(hist, dict) else None
        if isinstance(vl, list):
            val_loss_improved, val_loss_improved_note = _val_loss_improved_from_history(vl)
        else:
            val_loss_improved_note = "history.json missing val_loss list"
    else:
        val_loss_improved_note = "history.json not found"

    manual_flag = bool(manual_visual_ok) if manual_visual_ok is not None else False
    criteria_bools = [
        bool(val_loss_improved) if val_loss_improved is not None else False,
        bool(pass_margin),
        manual_flag,
    ]
    criteria_pass_count = sum(1 for x in criteria_bools if x)
    pass_relaxed = criteria_pass_count >= 2

    analysis = {
        "cnn_val": {k: v for k, v in cnn_metrics.items() if not isinstance(v, np.ndarray) and k != "patch_names"},
        "best_scalar_val_pearson": best_scalar if scalar_pearsons else None,
        "thresholds": {
            "pearson_min": pearson_min,
            "spearman_min": spearman_min,
            "pearson_margin": pearson_margin,
        },
        "verdict": {
            "pass_pearson": bool(pass_pearson),
            "pass_spearman": bool(pass_spearman),
            "pass_margin": bool(pass_margin),
            "pass_overall": bool(pass_pearson and pass_spearman and pass_margin),
        },
        "verdict_task2_relaxed": {
            "description": (
                "Pass if at least 2 of: val_loss_improved (from history.json early vs late val loss), "
                "pass_margin (CNN val Pearson > best_scalar + margin), "
                "manual_visual_ok (human-only: set true in JSON or pass manual_visual_ok= after reviewing "
                "highest_pred.txt / lowest_pred.txt vs Task 1 semantics)."
            ),
            "val_loss_improved": val_loss_improved,
            "val_loss_improved_note": val_loss_improved_note,
            "pass_margin": bool(pass_margin),
            "manual_visual_ok": manual_flag,
            "criteria_pass_count": criteria_pass_count,
            "pass_relaxed": bool(pass_relaxed),
        },
    }
    if baselines is not None:
        analysis["scalar_baselines"] = baselines

    save_json(analysis, out / "analysis.json")

    print("\n=== Task 2 Acceptance ===")
    print(f"  CNN val Pearson  = {cnn_metrics['pearson']:.4f}  (threshold > {pearson_min}): {'PASS' if pass_pearson else 'FAIL'}")
    print(f"  CNN val Spearman = {cnn_metrics['spearman']:.4f}  (threshold > {spearman_min}): {'PASS' if pass_spearman else 'FAIL'}")
    if scalar_pearsons:
        print(f"  Best scalar Pearson = {best_scalar:.4f}")
        print(f"  CNN - best scalar   = {cnn_metrics['pearson'] - best_scalar:+.4f}  (need > +{pearson_margin}): {'PASS' if pass_margin else 'FAIL'}")
    print(f"  OVERALL (strict): {'PASS' if analysis['verdict']['pass_overall'] else 'FAIL'}")
    vr = analysis["verdict_task2_relaxed"]
    vli = vr["val_loss_improved"]
    vli_str = "PASS" if vli is True else ("FAIL" if vli is False else "N/A")
    print(f"  Relaxed 2-of-3: val_loss_improved={vli_str}, pass_margin={'PASS' if pass_margin else 'FAIL'}, manual_visual_ok={'PASS' if manual_flag else 'FAIL (human)'}")
    print(f"  OVERALL (relaxed): {'PASS' if vr['pass_relaxed'] else 'FAIL'}  ({vr['criteria_pass_count']}/3 criteria)")

    return analysis
