## Efficiency (measured on validation loader, same batch size as training)

- Device: `cuda`
- Inference: mean wall time per patch (ms) over batches, after warmup.
- P0 and P5 share the same UNet; one timing reported for both.

| method | train_wall_clock_sec | notes_train_time | params_total | inference_ms_per_patch_mean | inference_peak_gpu_mb | deploy_inference |
| --- | --- | --- | --- | --- | --- | --- |
| P0_baseline | N/A | Not logged in artifacts; fill manually if needed. | 7763041 | 1.4733 | 286.91 | UNet only |
| P5_oracle_matched_trigger | N/A | Same architecture as P0; oracle weights use GT scores at train time only. | 7763041 | 1.4733 | 286.91 | UNet only (same as P0) |
| J3_joint_detach_false | N/A | Not logged in artifacts. | 7763298 | 1.1269 | 294.04 | Forward UNet; use logits output only — difficulty head can be omitted for deployment (tiny overhead if kept). |