# Manuscript assets (Task 3, frozen P0 / P5 / J3)

Suggested mapping for **main text** vs **supplementary**. Renumber to match your venue style.

## Main text (high priority)

| Asset | Path | Suggested label |
|-------|------|-----------------|
| Main results | [tables/main_results_final.md](tables/main_results_final.md) | **Table 1** — primary + secondary endpoints; multi-seed and cross-split aggregates |
| Qualitative (3 cases + strip) | [figures/qualitative_case_01.png](figures/qualitative_case_01.png), `02`, `03`, [figures/qualitative_panel_main.png](figures/qualitative_panel_main.png) | **Fig. 3** (or split: cases = Fig. 3, strip = Fig. 4) |
| J3 training dynamics | [figures/j3_training_dynamics.png](figures/j3_training_dynamics.png) | **Fig. 2** — phases (weight warmup → calib → pred weighting), loss, weights, val Dice |
| Case metadata | [figures/qualitative_case_manifest.json](figures/qualitative_case_manifest.json) | Caption / reproducibility |

## Supplementary (strongly recommended)

| Asset | Path | Notes |
|-------|------|--------|
| Mechanism ablations | [tables/ablation_mechanism.md](tables/ablation_mechanism.md) | J1 / J2 / J3 / calib0 / semantic control (`lambda_diff=0`) |
| Cross-split table | [tables/cross_split_summary.md](tables/cross_split_summary.md) | Per-entry splits; stress within-split deltas in caption |
| Calibration ablation curves | [figures/j3_calibration_ablation_curves.png](figures/j3_calibration_ablation_curves.png) | J3 vs J3_calib0 |
| Patch-level CSV + summary | [analysis/j3_patch_audit.csv](analysis/j3_patch_audit.csv), [analysis/j3_patch_audit_summary.json](analysis/j3_patch_audit_summary.json) | Full patch table |
| Patch scatter + thumbnails | [figures/j3_patch_scatter.png](figures/j3_patch_scatter.png), [figures/j3_patch_examples_panel.png](figures/j3_patch_examples_panel.png) | Supplementary figures |
| clDice | [tables/cldice_results.md](tables/cldice_results.md), [cldice_summary.json](cldice_summary.json) | Topology-oriented metric; threshold 0.5 |
| Efficiency | [tables/efficiency_table.md](tables/efficiency_table.md) | Params, inference ms/patch, peak GPU MB; train wall clock N/A |
| Training dynamics meta | [figures/training_dynamics_summary.json](figures/training_dynamics_summary.json) | Phase epoch ranges; whether per-epoch Pearson is present |
| Bootstrap + multi-seed | `outputs/multiseed_analysis.json` | Paired bootstrap CI (J3 vs P0 on hard patches); per-seed rows |
| Fixed-threshold robustness | Run `python scripts/multiseed_analysis.py` (Phase 3 section of script output) or document thresholds from `threshold_metrics.json` | Supplementary table |

## Deferred (plan P2)

- `lambda_diff` sensitivity grid — not run (see [writing_support_notes.md](writing_support_notes.md)).
- Focal / OHEM baseline — not run; discuss conceptually if needed.

## Method overview schematic (Fig. 1)

- No new schematic generated in this pass. Use a simple block diagram: encoder → seg head + difficulty head; training phases (warmup → calibration → predicted weighting with ramp). Reuse notation from `configs/task3_J3_joint_detach_false.yaml` / paper text.

## Regenerating assets

```text
python scripts/build_main_results_table.py
python -m src.analysis.j3_patch_audit
python scripts/paper_patch_audit_figures.py
python scripts/paper_qualitative_panels.py
python scripts/plot_j3_training_dynamics.py
python scripts/paper_efficiency_table.py
python scripts/eval_cldice_task3.py
python scripts/build_ablation_table.py
python scripts/build_cross_split_table.py
```

**Note:** Per-epoch `val_sci_corr_pearson` in `history.json` requires training with an updated `task3_engine.py`. Until J3 is re-run, the correlation subplot shows a placeholder (see `training_dynamics_summary.json`).
