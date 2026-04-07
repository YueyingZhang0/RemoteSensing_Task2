## Task 3 main results (canonical patch / group-split)

Unified reporting convention: fixed threshold \(\tau=0.5\) for main endpoints.

| Method (paper) | Global Dice @0.5 | Hard Dice @0.5 | clDice Hard @0.5 | Primary Hard Dice | best τ | internal_id |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline (P0) | 0.8104 | 0.7310 | 0.8915 | 0.7347 | 0.3000 | P0_baseline |
| Oracle-weighted reference (P5) | 0.8108 | 0.7347 | 0.8973 | 0.7374 | 0.3000 | P5_oracle_matched_trigger |
| CDC (proposed; internal J3) | 0.8107 | 0.7318 | 0.9036 | 0.7324 | 0.4000 | J3_joint_detach_false |
| Attention U-Net baseline | 0.8052 | 0.7260 | 0.8907 | 0.7272 | 0.3000 | attention_unet_baseline |
| Swin-UNet baseline | 0.8044 | 0.7215 | 0.8847 | 0.7240 | 0.3500 | swin_unet_baseline |
| nnU-Net v2 baseline | 0.8047 | 0.7242 | 0.9029 | 0.7285 | 0.3000 | nnunet_v2 |