# RemoteSensing — Task 2 / Task 3

## Task 3 主目标

**Task 3 的监督目标固定为 `sci_res2_norm`**（与 `configs/sce_probe.yaml` 中默认 `target_col` 一致）。

## 仓库与 `outputs/`

为控制体积，`outputs/` 下已忽略：

- 任意 `patches/` 目录（Task 1 导出的 patch 图）
- `*.pt` 检查点
- `*.png` / `*.gif` 等再生图

**建议纳入版本控制的产物**（若存在则会被跟踪）：`metrics.json`、`history.json`、`analysis.json`、`baseline_results.json`、`split.json`、`target_comparison_summary.json`、`patch_metadata.csv`、各类 `*.txt`（如 `highest_pred.txt`、`lowest_pred.txt`、`summary_stats.txt`）等。

完整训练与作图可在本地重新运行生成。

## Task 3（最小闭环）

- 固定 `sci_res2_norm`；数据与 Task 2 同源（patch 图 + `patches/masks` + CSV）。
- Baseline：`python -m src.main.train_task3 --config configs/task3.yaml --model baseline`
- UNet + SCE（bottleneck 分支 + 最浅 skip 软调制）：`python -m src.main.train_task3 --config configs/task3.yaml --model ours`
- 产物目录：`outputs/task3_unet_baseline/`、`outputs/task3_unet_sce/`（`best_model.pt`、`history.json`、`train_curves.png`、`val_metrics.json`、`hard_patch_metrics.json`）。

## Task 2 入口（摘要）

- 正式训练：`python -m src.main.train_sce_probe --config configs/sce_probe.yaml`
- 小样本过拟合诊断：`--debug-overfit`
- 全量诊断（每 epoch 相关等）：`--diagnostic-full`
- 多 target 对照：`python -m src.main.compare_task2_targets --config configs/sce_probe.yaml`
