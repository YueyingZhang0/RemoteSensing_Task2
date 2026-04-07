# nnU-Net v2 baseline (Task 3 patches) — external pipeline

This repo trains **in-house** models via `python -m src.main.train_task3`. **nnU-Net v2** uses its own preprocessing, plans, and trainer; compare it using the **same group split** and **same evaluation metrics** after you map predictions back to patches.

## 1. Export raw dataset

From the project root:

```bash
python scripts/export_task3_for_nnunet_v2.py --config configs/task3_v2_1_baseline.yaml
```

Optional: `--dataset_id 503` (default), `--out nnUNet_raw` (default). Output:

- `nnUNet_raw/Dataset503_TASK3/imagesTr/{case}_0000.png` … `_0002.png` (RGB)
- `labelsTr/{case}.png` (0/1)
- `dataset.json`, `splits_final.json` (train/val case IDs aligned with `split_json`)

### Windows / PATH 说明

`nnUNetv2_train.exe` 等往往在用户 `Python\Scripts` 下且**不在 PATH**。推荐一律用 **`python -m ...`**，不依赖 exe。

### 2. 设置环境变量（每个新终端都要设，或用下面脚本）

在 **vessel 仓库根目录** 下，三个路径指向本仓库内目录即可：

**PowerShell：**

```powershell
$root = "C:\Users\zhang\Desktop\vessel"   # 改成你的路径
$env:nnUNet_raw = "$root\nnUNet_raw"
$env:nnUNet_preprocessed = "$root\nnUNet_preprocessed"
$env:nnUNet_results = "$root\nnUNet_results"
New-Item -ItemType Directory -Force -Path $env:nnUNet_preprocessed, $env:nnUNet_results | Out-Null
```

### 3. 指纹 + 规划 + 预处理（只需成功一次）

模块名必须是 **`plan_and_preprocess_entrypoints`**（带 `s`），不是 `plan_and_preprocess_entry`：

```bash
python -m nnunetv2.experiment_planning.plan_and_preprocess_entrypoints -d 503 -c 2d --verify_dataset_integrity
```

完成后应存在：`nnUNet_preprocessed\Dataset503_TASK3\nnUNetPlans.json`。

### 4. 训练（fold 0 = 与导出的 `splits_final.json` 一致）

**方式 A（推荐）：** 自动设置环境并调用官方训练入口：

```bash
python scripts/nnunetv2_train_task3.py
```

无 GPU 时：

```bash
python scripts/nnunetv2_train_task3.py --device cpu
```

仅检查路径与依赖、不启动训练：

```bash
python scripts/nnunetv2_train_task3.py --dry-run
```

**方式 B：** 已在当前终端手动设置好 `nnUNet_*` 时：

```bash
python -m nnunetv2.run.run_training 503 2d 0
```

可选 `-device cuda`（默认）或 `-device cpu`。

训练输出在 `nnUNet_results` 下（由 nnU-Net 按数据集与配置创建子目录）。

### 5. 推理（导出全量 `imagesTr`，含概率 `.npz`）

训练结束后 `fold_0` 里应有 `checkpoint_final.pth`（或改用 `--chk checkpoint_best.pth`）。从仓库根目录：

```bash
python scripts/nnunetv2_predict_task3.py --out nnUNet_predictions/task3_fold0_prob
```

默认对 `nnUNet_raw/Dataset503_TASK3/imagesTr` 下**所有 case** 推理，并写 `--save_probabilities`（每个 case 一个 `{case_id}.npz`，内含 `probabilities`），便于与仓库内 **threshold sweep** 对齐。只要概率即可，不必单独导出「测试集子集」——下一步评测脚本会按 **val patch 列表** 过滤。

### 6. 评测：Hard Dice（threshold）+ clDice

```bash
python scripts/eval_nnunet_task3_predictions.py --pred-dir nnUNet_predictions/task3_fold0_prob
```

输出目录默认 `outputs/task3_nnunet_eval/`：

- `threshold_metrics.json`：全阈值 + **hard** 子集（与 `train_task3` 协议一致）
- `hard_patch_metrics.json`：hard patch 逐片指标
- `nnunet_cldice.json`：**clDice**（全局 + hard @0.5）

若只有 `.png` 无 `.npz`，脚本会给出警告；阈值扫描在单一概率下退化，建议重新推理并保留概率。

## 2. Fair comparison (paper checklist)

- **Same split** as `split_info.json` / export (group-wise train vs val).
- **Same primary metrics** as this repo: patch Dice / recall @ 0.5, hard subset, threshold sweep / best hard threshold — run nnU-Net predictions through **your existing evaluation** if possible, or document nnU-Net’s internal validation vs your `val_metrics.json` protocol.
- **Do not claim completed nnU-Net results** in the manuscript until numbers are produced and logged under `outputs/...` like other runs.

## 3. Planned vs completed

Formal nnU-Net v2 numbers are **planned validation** until you store reproducible artifacts (config hash, `dataset.json`, checkpoint, exported metrics JSON) alongside P0/J3 runs.
