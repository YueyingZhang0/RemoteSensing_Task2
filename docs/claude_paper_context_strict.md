# Claude / AI writing context — strict facts (Task 3)

Use this file only for **wording discipline**. Do not invent experiments or numbers not present in `outputs/` and tables.

## 0. Scope

**Completed (closed-loop) evidence** comes from the Task 3 **patch / group-split** protocol: `sci_res2_norm` difficulty, hard subset, threshold sweep, same-split / multi-seed / cross-split, J1/J2/J3 chain, clDice supplements.

**Not completed** — only future / planned (do **not** state as done): whole-image DRIVE reruns; DRIVE→CHASE/HRF migration; **SA-UNet formal baseline**; Betti errors; Focal/OHEM baselines; OCTA/ROSE/FIVES/remote-sensing transfer.

**Main paper table**: treat `tables/main_results_task3.md` + `tables/main_results_task3.csv` + `paper_assets/main_results_task3.json` as the authoritative source for the canonical (patch / group-split) headline numbers and definitions. Do not copy numbers from memory.

## 1. One-line positioning

J3 is the best **deployable** strategy found under the current protocol: it slightly exceeds the strongest oracle-weighted reference on the **canonical** split, is **consistently** better than baseline under **same-split multi-seed**, stays **competitive without collapse** under split perturbations (**weak pass vs P5**), and gains come from **calibrated, end-to-end coupled** difficulty weighting — not from “most accurate predictor wins” or extra capacity alone.

## 2. Forbidden claims

1. Do **not** write that J3 **stable beats oracle / “beyond oracle”** globally: vs P5, canonical split is directional; multi-seed favors J3; cross-split is **weak** vs P5.
2. Do **not** say “more accurate difficulty predictor ⇒ better segmentation”; J1/J2/J3 show **regression accuracy ≠ segmentation utility**.
3. Do **not** say the frozen probe “learned topology”; low correlation, mild utility only.
4. Do **not** cite DRIVE/CHASE/HRF migration as current main evidence.
5. Do **not** treat clDice as replacing the **primary** hard-patch endpoint; it is **supportive**.

## 3. Numeric anchors (verify in repo before citing)

Pull authoritative values from `outputs/*/threshold_metrics.json`, `val_metrics.json`, `multiseed_analysis.json`, `cross_split_analysis.json`, `warmup_ablation.json`, `semantic_control_result.json`, and clDice summaries — do not paraphrase from memory.
