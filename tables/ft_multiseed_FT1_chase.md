## FT-1. CHASE_DB1 fine-tune (test metrics) by seed

Metrics are **test split** means (eval_external_thinvessel protocol @0.5), stored as external_test_* in fine-tune run dirs. Mean±std over available seeds only (typically n=2); interpret cautiously.

| Method | seed | Dice @0.5 | Recall @0.5 | clDice @0.5 | Hard Dice @0.5 (thin-vessel proxy) | run_dir |
| --- | --- | --- | --- | --- | --- | --- |
| P0 | 42 | 0.7861 | 0.7736 | 0.8586 | 0.6566 | outputs\ft_chase_p0_seed42 |
| P0 | 40 | 0.7791 | 0.7351 | 0.8541 | 0.6124 | outputs\ft_chase_p0_seed40 |
| CDC (J3) | 42 | 0.7854 | 0.7652 | 0.8714 | 0.7008 | outputs\ft_chase_j3_seed42 |
| CDC (J3) | 40 | 0.7736 | 0.7855 | 0.8580 | 0.7098 | outputs\ft_chase_j3_seed40 |
