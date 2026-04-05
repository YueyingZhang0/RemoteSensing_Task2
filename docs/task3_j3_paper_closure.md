# Task 3 — J3 主方法收口节点（论文阶段）

**Git 节点**：与本文件同批提交为「Task 3 主方法定稿 + 验证闭环」的里程碑；此前已包含 multi-seed、cross-split、warmup ablation、semantic control 等完整产物与脚本。

**阶段判断**：不再处于「找方向」阶段，进入 **定稿主方法 + 撰写论文** 阶段。

---

## 1. 最终主方法与对照

| 角色 | 标识 | 说明 |
|------|------|------|
| **主方法** | `J3_joint_detach_false` | `configs/task3_J3_joint_detach_false.yaml` |
| **Oracle 参考** | `P5_oracle_matched_trigger` | `configs/task3_v3_1_oracle_matched_trigger.yaml` |
| **Baseline** | `P0` | `configs/task3_v2_1_baseline.yaml` |

---

## 2. 可写入论文的核心结论（有数据支撑）

### 2.1 原始 split 上的主指标

- **J3** `primary_hard_dice_mean` ≈ **0.738578**
- **P5** ≈ **0.737434**
- **P0** ≈ **0.734678**

主判据：`hard_dice_mean` @ `best_hard_dice_threshold`（与既有 Task 3 协议一致）。

### 2.2 Same-split 多 seed（方向稳定）

三 seed（40 / 41 / 42）汇总量级（见 `outputs/multiseed_analysis.json` 等）：

- J3 均值高于 P5、P0；**对 P5、P0 均为 3/3 seeds 在 primary 上领先**。

### 2.3 相对 baseline 的 hard-patch 提升（统计）

- seed=42：**paired bootstrap** 于 hard patches 上，J3 相对 P0 的 mean diff 为正，**95% CI 不跨 0**（详见 `scripts/multiseed_analysis.py` 与已保存的 `multiseed_analysis.json`）。

### 2.4 Cross-split（弱通过、需措辞谨慎）

- 额外 split（`split_info_rs100.json` / `split_info_rs200.json`）上：**无崩溃**；J3 在 **unique split 层面** 对 P5 为 **2/3 领先、1/3 近乎持平**（平均差距 &lt; 0.001）；**aggregate 上 J3 仍略高于 P5/P0**。
- **hd@0.5**：J3 在 cross-split 条目上仍表现一致优势（见 `scripts/cross_split_analysis.py` 输出）。

**推荐英文表述**（稳健）：

> J3 slightly exceeds the best oracle reference on the original split and remains **robustly competitive** across additional split perturbations, **without collapse**; the margin over the oracle reference is **small and split-sensitive**.

避免：**「稳定碾压 oracle」**、**「所有 split 上普遍更强」**、除非有更强外部验证，避免标题级 **「Beyond the Oracle」**。

### 2.5 机制证据链（与主方法叙事一致）

1. **J1**：predictor 相关性可较高，但 segmentation 仅小幅变化 → **预测难 ≠ 有用加权**。
2. **J2**（`pred_weight_detach: true`）：segmentation 明显劣于 baseline → **detach 阻断端到端时，加权有害**。
3. **J3**（`pred_weight_detach: false`）：当前最优 → **与分割目标端到端对齐的 difficulty weighting 是关键**。
4. **Warmup / calibration ablation**：去掉 `joint_calib` 等价阶段后 primary 大幅下降 → **校准期是必要条件**（见 `configs/task3_J3_calib0.yaml` 与 `warmup_ablation.json`）。
5. **Semantic control**（`lambda_diff: 0`）：低于 baseline → 收益来自 **有意义的 difficulty-aware 信号**，而非「多一个头 / 噪声通道」（见 `configs/task3_J3_semantic_control.yaml` 与 `outputs/semantic_control_result.json`）。

**可凝练为一句机制主张**：

> Difficulty regression accuracy alone is not sufficient; **calibration-first**, **end-to-end coupled** predicted difficulty weighting is what yields useful segmentation gains.

---

## 3. 复现与脚本索引

| 内容 | 路径 |
|------|------|
| Multi-seed 跑批 | `scripts/run_multiseed.py` |
| Multi-seed 汇总 | `scripts/multiseed_analysis.py` |
| 新 split 生成 | `scripts/generate_new_splits.py` |
| Cross-split 跑批 | `scripts/run_cross_split.py` |
| Cross-split 汇总 | `scripts/cross_split_analysis.py` |
| Semantic control 报告 | `scripts/semantic_control_report.py` |
| J3 patch 审计 | `src/analysis/j3_patch_audit.py` |

---

## 4. 后续（论文外审常见追问，非本节点必做）

- 更大规模或 **held-out 域 / 数据集** 的外部验证。
- 与 P5 差距的 **更广 split / 更多 seed** 下的置信区间叙述。
- 失败例与 **calibration 阶段** 的可视化（曲线、权重分布）。

---

*本文件用于锁定「主方法 + 主张边界」；具体数值以各 run 目录下 `threshold_metrics.json` / `val_metrics.json` 为准。*
