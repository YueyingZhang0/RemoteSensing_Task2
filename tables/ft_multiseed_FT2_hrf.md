## FT-2. HRF fine-tune (test metrics) by seed

Metrics are **test split** means (eval_external_thinvessel protocol @0.5), stored as external_test_* in fine-tune run dirs. Mean±std over available seeds only (typically n=2); interpret cautiously.

| Method | seed | Dice @0.5 | Recall @0.5 | clDice @0.5 | Hard Dice @0.5 (thin-vessel proxy) | run_dir |
| --- | --- | --- | --- | --- | --- | --- |
| P0 | 42 | 0.7543 | 0.7610 | 0.8234 | 0.4874 | outputs\ft_hrf_p0_seed42 |
| P0 | 40 | 0.7631 | 0.7728 | 0.8320 | 0.5149 | outputs\ft_hrf_p0_seed40 |
| CDC (J3) | 42 | 0.7420 | 0.7364 | 0.8241 | 0.5140 | outputs\ft_hrf_j3_seed42 |
| CDC (J3) | 40 | 0.7555 | 0.7853 | 0.8323 | 0.5736 | outputs\ft_hrf_j3_seed40 |
