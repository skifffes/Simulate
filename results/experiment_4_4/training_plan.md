# 实验 4.4 训练计划（执行前冻结）

日期 2026-11-05；起点提交 `d3e2b68`。
本轮目标：用**受控 A/B 两组**检验"扩大独立原子示范 + 充分训练预算"是否改善 π₀.₅ 的 A90/B90 原子能力。

## 1. 两组定义

| | Group A（小数据基线） | Group B（扩充数据） |
| --- | --- | --- |
| 数据根 | `/scratch/lijh/exp44_export_A58` | `/scratch/lijh/exp43/export_v3` |
| train | **58**（A90 30 / B90 28，旧 4.1 数据，已剔除 v1.2 不合格的 3 条 B90） | **311**（A90 170 / B90 141） |
| val / test | 16 / 13（与 B 完全相同，冻结） | 16 / 13 |
| 训练窗口 | 2,404 | 12,967 |

两组从**同一原始基座**独立初始化（不使用历史 step1800 作为 A 的起点）。

## 2. 完全一致的训练配置

| 项 | 值 |
| --- | --- |
| 基座 | `pi05_droid`（`checkpoints/pi05_droid/params`） |
| OpenPI | 固定版本（commit `215abfb`） |
| LoRA | `gemma_2b_lora` + `gemma_300m_lora`，trainable 466.96M / 3403.4M (13.72%) |
| 优化器 | `create_optimizer(cfg.optimizer, cfg.lr_schedule)`，两次运行同一函数 |
| LR schedule | 官方 CosineDecay：warmup **1000**、peak **2.5e-5**、decay **30000**、decay_lr 2.5e-6 |
| batch | 4（accum 1，未改 effective batch） |
| updates | **12,000**（两组相同） |
| 动作表示 | Anchor-16（`a_k = A[t+k] − S[t]`），`anchor_reference=fixed` |
| loss | 官方 32 维 flow matching（`loss_dims=0`） |
| 归一化 | 官方 DROID norm_stats（未改） |
| 种子 | `seed=0`（两组相同） |
| 输入 | 两路 v3 相机 + 语言 prompt + 9D state（8D 有效） |
| 评价器 | `exp4.1-strict-v1.2`（未改阈值） |

## 3. checkpoint 保存

| 内容 | 步数 |
| --- | --- |
| LoRA 权重（1.73 GB/个） | 1200, 1800, 2400, 3600, 4800, 6000, 8000, 10000, 12000 |
| optimizer state + RNG（3.7 GB/个，用于真正续训） | 6000, 12000 |
| 合并模型（12.6 GB） | **不保存**；离线评测时按需合并、用完即删 |

## 4. 数据曝光（必须说明）

| | Group A | Group B |
| --- | --- | --- |
| 独立 episode | 58 | 311 |
| 训练窗口 | 2,404 | 12,967 |
| 每个窗口被采样次数（名义） | 12,000×4 / 2,404 ≈ **19.97** | 12,000×4 / 12,967 ≈ **3.70** |
| 名义 epoch | ≈ 19.97 | ≈ 3.70 |

→ **相同 optimizer updates ≠ 相同 epoch**：Group A 是"少数据多遍"，Group B 是"多数据少遍"。
报告中所有比较都必须带这一条。

## 5. checkpoint 选择规则（**在看到 test 之前冻结**）

只用 **val**（16 条，冻结）与 val 初态闭环：

1. 对每个 LoRA checkpoint，合并后在**固定 30 个 val 状态 × 4 个固定采样种子**上评估；
2. 主指标（按优先级）：
   - `A90`：抓取阶段（grasp_close/lift）方向余弦中位数、joint MAE、夹爪开度误差；
   - `B90`：contact/push/wall_onset 阶段方向余弦中位数、Δz 偏差、joint MAE；
   - 采样方差（同状态不同种子之间的 MAE 标准差）；
3. 若有明显方差（种子间差 > 组内中位 20%），扩展到 8–12 个种子复核；
4. 取"分技能最佳"：A90 用 A90 指标、B90 用 B90 指标，**不默认最后一步最好**；
5. 若两个 checkpoint 指标接近（差 <5%），优先选**采样方差更小**的；
6. 规则冻结后，才允许在 test 与物理闭环上使用选中的 checkpoint。

## 6. 物理闭环协议（与 4.2/4.3 完全一致）

- 初始状态来自 4.1 冻结记录（物体位姿 + 机器人构型），prompt 用原始 prompt；
- `execute_k=8`、45 tick、12 physics steps/tick、`max_joint_step_rad=0.30`、continuous 夹爪；
- 严格评价器 v1.2；不注入任何专家动作；
- 每条 rollout 保存逐子步 CSV（tick/k/q 误差/EE/物体/接触/墙接触）；
- 每组先跑 4 条（A90×2 + B90×2）验安全，再扩到 6–8 条。

## 7. 可能的中断与恢复

- 训练带 `--max-hours 9.5` 兜底；超时会保存 `opt_stepNNNNN.npz` 后退出；
- 恢复命令：`--init-lora <lora_step> --resume-opt <opt_step>`（已验证可真正续训，见 `smoke_train`）；
- 若 GPU 被抢占：保存当前 LoRA + opt state，记录实际 updates，标记 PARTIAL。
