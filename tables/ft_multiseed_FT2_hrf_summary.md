## FT-2 summary. HRF mean ± std by method

Metrics are **test split** means (eval_external_thinvessel protocol @0.5), stored as external_test_* in fine-tune run dirs. Mean±std over available seeds only (typically n=2); interpret cautiously.

| Dataset | Method | n_seeds | Dice mean | Dice std | Recall mean | Recall std | clDice mean | clDice std | Hard Dice mean | Hard Dice std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HRF all (manual1; FOV mask) | P0 | 2 | 0.7587 | 0.0044 | 0.7669 | 0.0059 | 0.8277 | 0.0043 | 0.5012 | 0.0137 |
| HRF all (manual1; FOV mask) | CDC (J3) | 2 | 0.7487 | 0.0067 | 0.7608 | 0.0245 | 0.8282 | 0.0041 | 0.5438 | 0.0298 |
