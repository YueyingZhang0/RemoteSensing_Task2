# Task 3 v2.1 closure (stage boundary before v3)

This note records the **intended interpretation** of the v2.1 matrix on the fixed split (`outputs/task3_split/split_info.json`) and seed 42. It is not a claim about a single “best” checkpoint.

## Metrics convention

- **Primary (hard patch):** `hard_dice_mean` at **`best_hard_dice_threshold`** from `threshold_metrics.json` (sweep over 0.30–0.70).
- **Fixed 0.5:** `val_metrics.json` / `per_threshold["0.50"]` for reporting and calibration checks.

## Conclusions (minimal matrix)

1. **True linear (`oracle_true_linear`, γ=1)** vs **baseline:** On the primary hard-Dice-at-best-threshold criterion, true linear did **not** beat the baseline anchor; at threshold 0.5, hard Dice can look slightly different—interpret as **calibration / threshold sensitivity**, not a uniform frontier shift.
2. **Quadratic (`oracle_quadratic`, γ=2)** vs **true linear:** Quadratic scored **higher on primary** in the recorded matrix; treat **oracle quadratic** as the **upper-bound oracle reference** for this setup.
3. **WeightedRandomSampler (hard mask = `sci_res2_norm > tau` only, no FOV):** Gains vs the winning weighting mode were **small**; not treated as a reliable win without further replication.
4. **Smart / FOV gate:** With metadata `fov_ratio` mostly ≥ prior `fov_min`, the gate often **never fired**; raising `fov_min` made smart differ from linear but **hurt** metrics—**FOV gating is not a promising prior** on this patch distribution.

## Next (v3)

- **Predicted difficulty** (probe / joint paths already sketched in config), **same split**, **preserve all v2.1 code paths** for regression and ablations.
