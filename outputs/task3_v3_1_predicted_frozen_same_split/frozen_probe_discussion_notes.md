# Frozen probe (P4): discussion notes (conservative)

This note summarizes Task 3 v3.1 frozen-probe weighting after aligning metrics with the v2.1 matrix, adding matched-sparsity oracle controls (P5/P6), and a patch-level audit. **Language is intentionally cautious**; these are single-split, single-seed runs.

## 1. Fixed threshold 0.5 (reporting convention)

On the held-out val split, **P4 (frozen probe)** shows modest gains vs **P0 (baseline)** on both overall and hard subsets at probability threshold 0.5:

- `val_dice_mean @ 0.5`: P4 is slightly higher than P0 (see `frozen_probe_summary.csv`).
- `hard_dice_mean @ 0.5`: P4 is slightly higher than P0.
- `hard_recall_mean @ 0.5`: P4 is higher than P0 (frozen probe training emphasized recall-friendly behavior in places).

This is **not** purely a calibration artifact at 0.5 alone: the same directional pattern appears when reading `val_metrics.json` / `hard_patch_metrics.json` for P4 vs P0.

## 2. Primary metric (best hard-Dice threshold on val)

Following the v2.1 convention, the **primary** segmentation metric is:

`hard_dice_mean` evaluated at **`best_hard_dice_threshold`** from `threshold_metrics.json`.

For **P4**:

- `best_hard_dice_threshold` = **0.35**
- `primary_hard_dice_mean` ≈ **0.7361**
- `hard_recall` at that threshold ≈ **0.758**

Compared to **P0** and **P2 (oracle quadratic)**, P4 is slightly **better** on this primary metric (see `frozen_probe_summary.json`). The margin is **small** (on the order of 1e-3).

## 3. Matched-sparsity oracle controls (P5 / P6)

### P5 — matched trigger rate (train)

**Goal:** Match the fraction of training patches where **GT** `sci_res2_norm > tau_oracle` to the fraction where **probe prediction > 0.6** on the training split (~10.5%). **No sampler, no FOV gate**, same `split_json` and seed as v2.1.

**Result (primary metric):** P5’s primary hard Dice (**~0.7374** at `best_hard_dice_threshold` = 0.30) is **not below** P4; it is **slightly higher** on this run.

**Interpretation:** This is **preliminary evidence** that a **GT oracle with matched “trigger sparsity”** can achieve **at least comparable** hard-Dice-at-best-threshold to frozen probe weighting. It **does not support** a strong claim that the probe identifies a **unique** set of patches for segmentation; it is **consistent with** a story where **similar sparsity of upweighting** matters.

### P6 — matched strength (train, best-effort)

**Goal:** Approximate the **pooled distribution** of per-sample training weights implied by probe predictions (via the same `oracle_sample_weights` map) using **GT scores** and tuned `(tau, alpha, max_weight)`.

**Limitation:** Under batch normalization of weights, a **GT-based** quadratic oracle could not reproduce the probe-implied pooled **std** (~0.012) in our grid; the calibrated P6 setting reflects a **best-effort** compromise (see `matched_oracle_calibration.json` and comments in `task3_v3_1_oracle_matched_strength.yaml`).

**Result:** P6 primary hard Dice (**~0.7351**) sits **between** P2 and P4, **not** clearly above P4.

**Interpretation:** P6 is a **weaker** control than P5 for the specific question “same sparsity”; still, it suggests that **moderate, sparse oracle reweighting** can land in a **similar performance band** without using probe predictions.

## 4. Patch-level audit (val, tau = 0.6)

Files: `frozen_patch_audit.csv`, `frozen_patch_groups_summary.json`.

**Trigger definition (analysis only):**

- `pred_above_tau`: probe output > 0.6  
- `oracle_above_tau`: GT `sci_res2_norm` > 0.6  

**Four groups (val, n=172):**

| Group        | n   | Comment (high level) |
|-------------|-----|----------------------|
| `pred_only` | 8   | Probe fires above 0.6 while GT does not; **largest mean** `delta_dice` and `delta_recall` vs baseline on this split. |
| `oracle_only` | 18 | GT hard by threshold but probe low; still **positive** mean delta vs baseline (per-patch dice/recall). |
| `both`      | 8   | Agreement; smaller mean delta than `pred_only`. |
| `neither`   | 138 | Bulk of val; small positive mean deltas. |

**Topology / geometry (means):** `pred_only` patches tend to have **lower** mean `fov_ratio` and **higher** mean `vessel_area` than `oracle_only` (see summary JSON). This is **descriptive only**; we do not claim causal structure.

**Representative images:** `patch_group_examples/{group}/` (top by |delta_dice|).

## 5. Provisional conclusion

1. **Fixed @ 0.5:** P4 is **slightly favorable** vs baseline on val/hard metrics; not obviously “calibration only.”

2. **Primary @ best hard threshold:** P4 is **slightly favorable** vs P0 and P2 on this split; margins are **small**.

3. **Matched controls:** **P5 ≥ P4** on the primary metric in this experiment. That **undercuts** a strong “probe-only benefit” narrative and is **consistent with** **weak / sparse reweighting** (and threshold selection) explaining much of the effect.

4. **Overall verdict (choose one):** **C — unclear / cannot yet distinguish** between a **prediction-specific** story and **conservative sparse reweighting**, with a **lean** toward **B (weak sparse regularization / sparsity-matched effects)** given P5. We **cannot** responsibly claim that **predicted difficulty beats oracle** in general; at most, **on this split**, frozen probe is **competitive** and **P5 shows matched-sparsity GT weighting can match or exceed it on the primary metric**.

---

*Artifacts: `frozen_probe_summary.json`, `frozen_probe_summary.csv`, `matched_oracle_calibration.json`, `frozen_patch_audit.csv`, `frozen_patch_groups_summary.json`, `patch_group_examples/`.*
