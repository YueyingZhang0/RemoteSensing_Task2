## Main results (Task 3, frozen protocol)

**Primary endpoint:** `hard_dice_mean` at **best_hard_dice_threshold** (selected on validation grid; same grid for all methods).

**Secondary endpoints:** `hard_dice_mean` at fixed threshold 0.5; `val_dice_mean` at 0.5.

**Canonical split:** `outputs/task3_split/split_info.json`, `random_state=42` (seed42 columns).

**Multi-seed:** mean ± std over seeds 40, 41, 42 on the **same** split (see `outputs/multiseed_analysis.json`).

**Cross-split:** mean ± std over run entries in `outputs/cross_split_analysis.json` (includes alternate splits and an extra seed on split B). Absolute primary values are **not** comparable across splits (hard subset changes); use within-split deltas for robustness claims. Aggregate means are reported for summary only.

| method | primary_hd_seed42 | hd_at_0_5_seed42 | vd_at_0_5_seed42 | best_hard_dice_th_seed42 | hard_recall_at_best_hd_seed42 | multi_seed_primary_hd_mean | multi_seed_primary_hd_std | multi_seed_hd_0_5_mean | multi_seed_hd_0_5_std | multi_seed_vd_0_5_mean | multi_seed_vd_0_5_std | multi_seed_best_hd_th_mean | multi_seed_hr_at_best_hd_mean | multi_seed_hr_at_best_hd_std | cross_split_primary_hd_mean | cross_split_primary_hd_std | cross_split_hd_0_5_mean | cross_split_hd_0_5_std | cross_split_vd_0_5_mean | cross_split_vd_0_5_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P0_baseline | 0.734678 | 0.730991 | 0.810364 | 0.300000 | 0.744569 | 0.734714 | 0.000459 | 0.733118 | 0.001756 | 0.810442 | 0.000453 | 0.366667 | 0.751728 | 0.007132 | 0.712820 | 0.014232 | 0.710539 | 0.013133 | 0.788711 | 0.013309 |
| P5_oracle_matched_trigger | 0.737434 | 0.734693 | 0.810798 | 0.300000 | 0.750809 | 0.735359 | 0.001530 | 0.733498 | 0.001114 | 0.811176 | 0.000339 | 0.383333 | 0.749083 | 0.003955 | 0.714688 | 0.015382 | 0.711518 | 0.015173 | 0.791427 | 0.011326 |
| J3_joint_detach_false | 0.738578 | 0.736539 | 0.811066 | 0.350000 | 0.764107 | 0.736125 | 0.001801 | 0.735142 | 0.001006 | 0.811172 | 0.001100 | 0.433333 | 0.755991 | 0.006001 | 0.714993 | 0.016135 | 0.713718 | 0.015710 | 0.790626 | 0.012294 |