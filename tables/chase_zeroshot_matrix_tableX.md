## Table X — CHASE_DB1 zero-shot (thin-vessel primary)

Primary metric for this matrix: **Hard Dice @0.5** (thin-vessel proxy mean). Dice / Recall / clDice @0.5 are secondary and were fixed a priori.

nnU-Net v2 row: standard nnU-Net preprocessing / resampling / normalization at inference; probabilities resized to CHASE GT size before @0.5 (not the raw sliding-window PyTorch protocol).

| Method | Hard Dice @0.5 (thin-vessel proxy) | Dice @0.5 | Recall @0.5 | clDice @0.5 | internal_id |
| --- | --- | --- | --- | --- | --- |
| P0 U-Net baseline (zero-shot) | 0.2337 | 0.1786 | 0.1029 | 0.5332 | p0_baseline_zeroshot_chase |
| CDC / J3 joint (zero-shot) | 0.3079 | 0.2690 | 0.1688 | 0.5409 | cdc_j3_zeroshot_chase |
| Attention U-Net baseline (zero-shot) | 0.4466 | 0.3908 | 0.2984 | 0.5696 | attention_unet_baseline_zeroshot_chase |
| Swin-UNet baseline (zero-shot) | 0.3794 | 0.3221 | 0.2450 | 0.5318 | swin_unet_baseline_zeroshot_chase |
| nnU-Net v2 (zero-shot) | 0.6811 | 0.6259 | 0.5906 | 0.7659 | nnunet_v2_zeroshot_chase |
