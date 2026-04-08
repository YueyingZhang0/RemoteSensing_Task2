## FT-1 summary. CHASE mean ± std by method

Metrics are **test split** means (eval_external_thinvessel protocol @0.5), stored as external_test_* in fine-tune run dirs. Mean±std over available seeds only (typically n=2); interpret cautiously.

| Dataset | Method | n_seeds | Dice mean | Dice std | Recall mean | Recall std | clDice mean | clDice std | Hard Dice mean | Hard Dice std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CHASE_DB1 (1st observer) | P0 | 2 | 0.7826 | 0.0035 | 0.7543 | 0.0192 | 0.8563 | 0.0023 | 0.6345 | 0.0221 |
| CHASE_DB1 (1st observer) | CDC (J3) | 2 | 0.7795 | 0.0059 | 0.7754 | 0.0101 | 0.8647 | 0.0067 | 0.7053 | 0.0045 |
