# 实验 4.5：Action Expert-only 全参数微调 vs LoRA（311 条专家示范）

本轮唯一变量是**微调参数范围**：4.4 的 LoRA（低秩） vs 4.5 的 Action Expert 全参数。
数据、batch、seed、Anchor-16、DROID normalization、严格评价器全部保持不变。

## 目录

| 文件 | 内容 |
| --- | --- |
| `training_plan.md` | 假设、受控变量、预算、选择规则（先冻结） |
| `training_config.json` | 完整配置（数据/模型/优化器/评测） |
| `parameter_partition.json` | 51 个张量的可训练/冻结分区与参数量 |
| `weight_loading_audit.md` | Gate 1：权重确实来自 pi05_droid（含 adaRMS 0→0.00968 证据） |
| `smoke_test.md` | Gate 2（冻结/梯度）、Gate 3（A40 可行性）、Gate 4（逐位一致的续训） |
| `memory_benchmark.json` | 峰值显存 34.4 GB、JIT 247 s（冷）/16.6 s（热）、1.05 step/s |
| `training_curve.csv` | 逐 20 步的 loss/grad/吞吐 |
| `checkpoint_manifest.json` | 7 个 checkpoint + optimizer state 的 sha256 |
| `checkpoint_selection.md` | 联合选择规则与结果（A90+B90 同一模型） |
| `offline_eval_results.json` | 逐 checkpoint 的离线指标（含分阶段） |
| `paired_rollout_results.csv` | 16 条 val episode × 3 seed × 2 模型的配对闭环 |
| `failure_analysis.md` | 失败模式与原始证据 |
| `final_report.md` | 最终报告（11 问） |
| `artifact_manifest.json` | 服务器路径 + SHA256 |

## 复现

```bash
# Gate 2/3 审计（参数分区 + 6 步梯度审计 + ckpt/opt）
python scripts/exp45/gate2_audit.py --steps 6

# 正式训练（12,000 updates，batch 4，GPU0）
bash server/exp45/train_ae.sh

# 离线评测（7 个 checkpoint × 30 val 状态 × 4 seeds）
bash server/exp45/eval_driver_ae.sh

# 联合选择
python scripts/exp45/joint_select.py

# 配对闭环（AE 与 LoRA 各 16 episode × 3 seed）
bash server/exp45/run_paired.sh ae <STEP> 3
bash server/exp45/run_paired.sh lora <STEP> 3
python scripts/exp45/aggregate_rollouts.py
```
