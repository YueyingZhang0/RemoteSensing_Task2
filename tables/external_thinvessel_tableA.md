## Table A. External-style evaluation (dataset-agnostic thin-vessel hard proxy)

| Dataset | Method | Setting | Dice @0.5 | Recall @0.5 | clDice @0.5 | Hard Dice @0.5 (thin-vessel proxy) | n_train | n_val | n_test | seed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DRIVE (training subset only in repo) | P0 | re-eval | 0.8304 | 0.8052 | 0.9249 | 0.8464 | 0 | 0 | 20 | N/A |
| DRIVE (training subset only in repo) | CDC (J3) | re-eval | 0.8381 | 0.8387 | 0.9397 | 0.8759 | 0 | 0 | 20 | N/A |
| CHASE_DB1 (1st observer) | P0 | zero-shot | 0.1786 | 0.1029 | 0.5332 | 0.2337 | 8 | 4 | 16 | 42 |
| CHASE_DB1 (1st observer) | CDC (J3) | zero-shot | 0.2690 | 0.1688 | 0.5409 | 0.3079 | 8 | 4 | 16 | 42 |
| CHASE_DB1 (1st observer) | P0 | fine-tuned | 0.7861 | 0.7736 | 0.8586 | 0.6566 | 8 | 4 | 16 | 42 |
| CHASE_DB1 (1st observer) | CDC (J3) | fine-tuned | 0.7854 | 0.7652 | 0.8714 | 0.7008 | 8 | 4 | 16 | 42 |
| HRF all (manual1; FOV mask) | P0 | zero-shot | 0.0292 | 0.0155 | 0.4681 | 0.0180 | 10 | 5 | 30 | 42 |
| HRF all (manual1; FOV mask) | CDC (J3) | zero-shot | 0.0696 | 0.0384 | 0.4600 | 0.0598 | 10 | 5 | 30 | 42 |
| HRF all (manual1; FOV mask) | P0 | fine-tuned | 0.7543 | 0.7610 | 0.8234 | 0.4874 | 10 | 5 | 30 | 42 |
| HRF all (manual1; FOV mask) | CDC (J3) | fine-tuned | 0.7420 | 0.7364 | 0.8241 | 0.5140 | 10 | 5 | 30 | 42 |