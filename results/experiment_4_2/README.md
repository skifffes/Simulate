# 实验 4.2：π₀.₅ 原子技能联合 LoRA 微调

**状态：训练链路 PASS，原子技能未学会（0/8 严格成功）。** 详见
[`reports/experiment_4_2_report.md`](reports/experiment_4_2_report.md)。

- Simulate 基准 `b1b1c41`；OpenPI `215abfb`；基座 `pi05_droid`；相机 v3；评价器 `exp4.1-strict-v1.2`
- 训练：LoRA（`gemma_2b_lora` + `gemma_300m_lora`），可训练 466.96 M / 3403.4 M = **13.72%**
- 步数 **901**（501 + 续训 400），batch 4，**0.44–0.55 step/s**，loss 1.16 → **0.027**（val 0.0265）
- 闭环：4 条 A90 + 4 条 B90（两个 checkpoint 各 2+2）→ **0 strict success**；
  step501 完全不接触方块，step901 开始接触（69/21/0/22 物理步）但抓取/推动方向不对

## 目录

| 路径 | 内容 |
| --- | --- |
| `environment_manifest.json` | 环境、checkpoint、相机、数据集、split、**lerobot 缺失的阻塞说明** |
| `training_data_contract.md` | Gate 2：动作语义链、32D padding/无 mask、horizon、episode 尾部、夹爪、20 Hz |
| `normalization_decision.md` | DROID stats vs train-split 统计、clip 比例、决策依据 |
| `normalization_stats_check.json` | train split 的分位数与越界比例（机器可读） |
| `train_config.yaml` | 冻结训练配置（模型/LoRA/优化器/批大小/GPU） |
| `training_speed_benchmark.json` | step 耗时、吞吐、compile 时间、等效 epoch |
| `training_logs/train_step501.log` | 第一段训练逐步日志 |
| `training_logs/train_step901.log` | 续训逐步日志（含 val loss） |
| `checkpoint_manifest.json` | 两个 checkpoint 的 md5、大小、可加载方式、rollout 结果 |
| `action_compare_baseline_vs_finetuned.json` | Gate 4：同观测下基线 vs 微调的动作差 |
| `rollouts_step501/`, `rollouts_step901/` | Gate 5：每条 rollout 的严格评价、接触步、动作日志 |
| `atomic_eval_summary.json` | 汇总（含总体判定） |
| `failure_analysis.md` | 为什么 loss 降了但成功率为 0（第一版：动作尺度诊断） |
| `diagnostics_p0.md` | **P0 诊断（review 后）**：专家观测上的幅度/方向/夹爪对比 + loss 归因 + 裁剪统计；修正了"幅度是唯一根因" |
| `action_diagnostics_p0.json` | 上述诊断的逐帧原始数据（10 帧 × 3 个模型） |

## 数据集与 split（复用 4.1，未重新采集）

| split | A90 | B90 | 帧 |
| --- | --- | --- | --- |
| train | 30 | 31 | 10 882 |
| val | 10 | 6 | 2 941 |
| test | 5 | 8 | 2 261 |

`exp4.1-split-v1` 冻结；训练窗口 = 2521（horizon 16、stride 4）。

## 复现

```bash
python scripts/exp42/export_data.py --data data/exp41 --out /scratch/lijh/exp42/data
bash server/exp42/run_train.sh train --steps 501 --batch 4 --save-every 100 --export-params
python scripts/exp42/export_params_from_lora.py --lora <npz> --out <dir>/params
python scripts/exp42/compare_actions.py --ckpt <dir>
bash server/exp42/run_eval.sh <dir> 8124 2 45 1
```

## 结论一句话

**π₀.₅ 的 LoRA 训练、checkpoint 合并、推理加载、20 Hz 闭环与严格物理评价全部真实跑通；
但 901 步（1.43 epoch）后在 8 条 rollout 上仍未学会 A90/B90——失败集中在动作幅度比训练分布大 4–10 倍，
导致机械臂无法稳定接近/抓取/定向推动。**
