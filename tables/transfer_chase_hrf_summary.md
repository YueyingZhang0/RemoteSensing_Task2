## External transfer (zero-shot; whole-image)

Fixed threshold \(\tau=0.5\). Sliding-window inference: patch=128, stride=64.

| dataset | Method (paper) | Dice @0.5 (mean) | Recall @0.5 (mean) | clDice @0.5 (mean) | n_images | internal_id |
| --- | --- | --- | --- | --- | --- | --- |
| CHASE_DB1 (1st observer) | Baseline (P0) | 0.1576 | 0.0897 | 0.4998 | 28 | P0_baseline |
| CHASE_DB1 (1st observer) | CDC (proposed; internal J3) | 0.2577 | 0.1608 | 0.5263 | 28 | J3_joint_detach_false |
| HRF (all; manual1; FOV mask if present) | Baseline (P0) | 0.0299 | 0.0161 | 0.4636 | 90 | P0_baseline |
| HRF (all; manual1; FOV mask if present) | CDC (proposed; internal J3) | 0.0713 | 0.0397 | 0.4520 | 90 | J3_joint_detach_false |