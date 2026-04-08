## FT-3. Fine-tune minus zero-shot (test split, same protocol)

Metrics are **test split** means (eval_external_thinvessel protocol @0.5), stored as external_test_* in fine-tune run dirs.

| Dataset | Method | seed | ΔDice | ΔRecall | ΔclDice | ΔHard Dice (thin-vessel proxy) | run_dir |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CHASE_DB1 (1st observer) | P0 | 42 | 0.6075 | 0.6707 | 0.3254 | 0.4228 | outputs\ft_chase_p0_seed42 |
| CHASE_DB1 (1st observer) | P0 | 40 | 0.6005 | 0.6322 | 0.3209 | 0.3787 | outputs\ft_chase_p0_seed40 |
| CHASE_DB1 (1st observer) | CDC (J3) | 42 | 0.5164 | 0.5964 | 0.3305 | 0.3929 | outputs\ft_chase_j3_seed42 |
| CHASE_DB1 (1st observer) | CDC (J3) | 40 | 0.5046 | 0.6166 | 0.3172 | 0.4018 | outputs\ft_chase_j3_seed40 |
| HRF all (manual1; FOV mask) | P0 | 42 | 0.7251 | 0.7455 | 0.3553 | 0.4694 | outputs\ft_hrf_p0_seed42 |
| HRF all (manual1; FOV mask) | P0 | 40 | 0.7339 | 0.7573 | 0.3640 | 0.4969 | outputs\ft_hrf_p0_seed40 |
| HRF all (manual1; FOV mask) | CDC (J3) | 42 | 0.6724 | 0.6980 | 0.3641 | 0.4542 | outputs\ft_hrf_j3_seed42 |
| HRF all (manual1; FOV mask) | CDC (J3) | 40 | 0.6859 | 0.7469 | 0.3723 | 0.5139 | outputs\ft_hrf_j3_seed40 |
