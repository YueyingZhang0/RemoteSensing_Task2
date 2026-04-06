# Writing support notes (Task 3 paper)

## 1. Main claim

1. **Direct soft modulation of skip features is unstable** for this setting (prior Task 3 / SCE line; cite your earlier failure mode). The viable path is **explicit sample reweighting** driven by difficulty, not ad-hoc feature gating.

2. **Oracle difficulty weighting** (P5, matched-trigger quadratic on GT `sci_res2_norm`) is a **strong, non-deployable upper reference**: it shows that reweighting hard patches can help when the difficulty signal is correct.

3. **Difficulty regression accuracy alone does not guarantee useful weighting.** J1 shows a predictor can correlate with GT difficulty while segmentation gains remain modest; **detach=true** (J2) can even **hurt** segmentation despite reasonable correlations—prediction quality ≠ training-time usefulness for weighting.

4. **Calibration-first, end-to-end joint difficulty weighting** (J3, `pred_weight_detach=false`) is the **most effective deployable strategy** found: a **calibration phase** where the head learns under uniform seg loss + aux regression, followed by **predicted weighting with ramp** and **gradients through the head** into the weighted seg loss.

5. The **primary goal** is **hard-patch / topology-fragile prioritization** (hard Dice at a threshold chosen on val), **not** merely maximizing global validation Dice. Secondary metrics (`hd@0.5`, `vd@0.5`) and clDice provide context but do not replace the primary endpoint.

**Stable one-line positioning:** J3 slightly exceeds the best oracle reference on the canonical split, remains competitive across split perturbations without collapse, and the gain is **not** explained by an extra head alone (semantic control) or by skipping calibration (calib0 ablation).

---

## 2. Evidence chain (matches figures/tables)

| Claim | Evidence |
|-------|-----------|
| J3 best on canonical primary | [tables/main_results_final.md](tables/main_results_final.md) seed-42 row; multi-seed means |
| Not single-seed luck | Same table: multi-seed mean ± std; `outputs/multiseed_analysis.json` |
| J3 vs baseline on hard patches | Bootstrap CI on hard patches (J3 − P0) excludes 0 — `multiseed_analysis.json` |
| Oracle reference | P5 row in main table; cross-split rows in [tables/cross_split_summary.md](tables/cross_split_summary.md) |
| Cross-split cautious claim | Aggregate means in main table + verdict in `cross_split_analysis.json` (2/3 unique splits vs P5; split B ~ tie) |
| Calibration necessary | [tables/ablation_mechanism.md](tables/ablation_mechanism.md) J3 vs J3_calib0; [figures/j3_calibration_ablation_curves.png](figures/j3_calibration_ablation_curves.png) |
| End-to-end matters | Ablation table J2 vs J3 |
| Semantic signal matters | J3 vs `lambda_diff=0` (semantic control) in ablation table |
| Qualitative hardest-patch story | [figures/qualitative_case_01.png](figures/qualitative_case_01.png)–`03` + [figures/qualitative_case_manifest.json](figures/qualitative_case_manifest.json); failure case included |
| Training phases | [figures/j3_training_dynamics.png](figures/j3_training_dynamics.png) + [figures/training_dynamics_summary.json](figures/training_dynamics_summary.json) |
| Topology-oriented metric | [tables/cldice_results.md](tables/cldice_results.md) — supports narrative but is **not** the primary endpoint |
| Efficiency | [tables/efficiency_table.md](tables/efficiency_table.md) — J3 deploy: forward UNet, **logits only**; difficulty head optional at inference |

---

## 3. Limitations (use in Discussion)

- **Small data / few val groups:** Group splits with only four validation IDs make **absolute** cross-split primary metrics incomparable; emphasize **within-split** rankings and **absence of collapse**.
- **Margin vs P5:** Advantage over matched oracle is **small**; J3 vs P5 bootstrap on hard patches **not** significant on canonical split—avoid “crushes oracle” language.
- **clDice:** Implemented as **hard** binarized clDice with fixed dilation; ranking can differ from Dice; report as **supplementary** topology signal.
- **Per-epoch Pearson:** Current saved `history.json` may lack `val_sci_corr_pearson` until re-training with updated logging; final correlation still in `difficulty_stats.json`.
- **Train wall-clock:** Not logged in artifacts; table marks **N/A** unless measured manually.
- **External validation:** No out-of-dataset test; generalization claims should be cautious.

---

## 4. Figure / table placement suggestions

- **Fig. 1:** Method + phase diagram (warmup → calib → pred weighting + ramp). *Not auto-generated.*
- **Fig. 2:** [figures/j3_training_dynamics.png](figures/j3_training_dynamics.png) — tie caption to phase bands in `training_dynamics_summary.json`.
- **Fig. 3:** Qualitative panel — use `qualitative_case_01–03` and/or [figures/qualitative_panel_main.png](figures/qualitative_panel_main.png); cite manifest for patch IDs and Dice.
- **Table 1:** [tables/main_results_final.md](tables/main_results_final.md) — footnote primary vs secondary endpoints.
- **Table 2 (Supp.):** [tables/ablation_mechanism.md](tables/ablation_mechanism.md).
- **Table 3 (Supp.):** [tables/cross_split_summary.md](tables/cross_split_summary.md) + short delta narrative from `cross_split_analysis.json`.
- **Supp. figs:** [figures/j3_patch_scatter.png](figures/j3_patch_scatter.png), [figures/j3_patch_examples_panel.png](figures/j3_patch_examples_panel.png), calibration ablation PNG, clDice table.

---

## Answers for the “stop condition” checklist

1. **Main-text ready:** Main results table; qualitative figures; J3 training dynamics; manifest JSON.  
2. **Supplementary:** Ablation table, cross-split table, patch audit CSV/JSON + scatter/examples, clDice, efficiency, bootstrap JSON, calibration ablation figure.  
3. **clDice vs narrative:** On this run, J3 ≥ P5 ≥ P0 on global and hard-subset clDice @0.5 — **consistent with** hard-patch prioritization but **not** a substitute for the primary hard-Dice endpoint.  
4. **Qualitative vs claim:** Case 1 is a clear hard-patch win; Case 2 shows J3 ≈ P5; Case 3 is a **hard-patch failure** (J3 below P0)—use to bound claims (“not uniform improvement”) while showing non-trivial wins exist on difficult patches.
