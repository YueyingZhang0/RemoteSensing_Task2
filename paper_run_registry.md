# Paper run registry (frozen)

This file **freezes** the canonical (Task 3 patch / group-split) runs used as the paper's gold-standard evidence.

**Rules**

- These run folders and their files are treated as **read-only**: do not overwrite, re-run into the same directory, or edit files in-place.
- External transfer / fine-tune experiments must write to **new** `outputs/ext_*` / `outputs/ft_*` directories and must not mix into the canonical main table.

## Canonical runs (patch / group-split protocol)

### P0 (baseline)

- **output_dir**: `outputs/task3_v2_1_baseline_same_split`
- **config_used**: `outputs/task3_v2_1_baseline_same_split/config_used.yaml`
- **git_commit (recorded at run time)**: `6b09e0478e16d6b8a8ff1079bb246f2f69638ae1`
- **timestamp (recorded at run time)**: `2026-04-04T11:00:47.164907+00:00`
- **split_json**: `outputs/task3_split/split_info.json`

### P5 (oracle-weighted reference)

- **output_dir**: `outputs/task3_v3_1_oracle_matched_trigger_same_split`
- **config_used**: `outputs/task3_v3_1_oracle_matched_trigger_same_split/config_used.yaml`
- **git_commit (recorded at run time)**: `321ec946f46a569e46a265e3183a66b7a5af9f67`
- **timestamp (recorded at run time)**: `2026-04-04T23:33:48.257256+00:00`
- **split_json**: `outputs/task3_split/split_info.json`

### J3 (CDC; proposed method)

- **output_dir**: `outputs/task3_J3_joint_detach_false`
- **config_used**: `outputs/task3_J3_joint_detach_false/config_used.yaml`
- **git_commit (recorded at run time)**: `d078976e3acfc7d6f0d7797e329f55167c27b94a`
- **timestamp (recorded at run time)**: `2026-04-06T00:57:50.653426+00:00`
- **split_json**: `outputs/task3_split/split_info.json`

## Registry last updated

- **repo HEAD when updated**: `d1f7148100f4809e49e85482244e178902072918`

