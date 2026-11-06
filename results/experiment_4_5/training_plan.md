# 实验 4.5 训练计划（执行前冻结）

日期 2026-11-06；基准提交 `060dadf`；OpenPI `215abfb`。

## 1. 待检验假设

**解除 Action Expert 的低秩更新约束，可能改善 π₀.₅ 的动作生成质量与真实闭环能力。**
这是假设，不是结论；本轮结果可能支持也可能否定。

## 2. 唯一变量

| 项 | 4.4 LoRA（对照） | 4.5 Action Expert-only |
| --- | --- | --- |
| 微调范围 | `gemma_2b_lora` + `gemma_300m_lora`（LoRA 张量） | Action Expert **全部非 LoRA 权重** + 动作投影 |
| 可训练参数 | 467.0 M（13.72%） | **430.1 M（12.83%）** |
| 数据集 | export_v3 311 条 | 同 |
| batch / seed | 4 / 0 | 同 |
| LR / warmup / decay | 官方 2.5e-5 / 1000 / 30000 | 同 |
| 动作表示 / normalization | Anchor-16, fixed / DROID | 同 |
| 评价器 | `exp4.1-strict-v1.2` | 同 |

**未改**：动作标签、相机、数据增强、控制器、夹爪单位、成功阈值、horizon、loss 形式。
**未加入**任何 A→B 组合示范。

## 3. 可训练/冻结分区

- 可训练：`PaliGemma/llm/**/*_1`（layers 的 attn/mlp/两个 adaRMSNorm、final_norm_1）
  + `action_in_proj` + `action_out_proj` + `time_mlp_in` + `time_mlp_out` = **19 个张量、430.1 M**；
- 冻结：`PaliGemma/img/*`（SigLIP 22 张量）+ `PaliGemma/llm` 中非 `_1` 的主干与 embedding（10 张量）
  = **32 个张量、2923.3 M**。
- 详细 51 行见 `parameter_partition.json`；Gate 2 实测冻结张量 0/32 改变、专家 19/19 改变。

## 4. 预算与保存

- **12,000 optimizer updates**（batch 4，effective batch = 4，未使用梯度累积）；
- 保存 LoRA 风格 npz（仅 trainable）：1200 / 2400 / 4000 / 6000 / 8000 / 10000 / 12000；
- optimizer state（3.44 GB）：6000 / 12000；
- `--max-hours 5.0` 兜底：超时会保存 opt state 再退出，可续训（Gate 4 已验证逐位一致）。

## 5. 数据曝光

311 条 train、12,967 个 anchor-16 窗口，12,000×4 采样 → **名义 3.70 epoch**。
与 4.4 Group B 完全相同，因此"相同 updates = 相同曝光"。

## 6. checkpoint 选择规则（在看到 test 之前冻结）

**联合规则（同一模型必须同时具备 A90/B90 能力）**：

1. A90 用抓取/保持阶段（`A90:gripper_closed`）的 joint MAE（专家运动≈0，不用 cos）；
2. B90 用主动推动阶段（`B90:motion_high`）的方向余弦；
3. 组内各自排名，`joint_score = rank_A90 + rank_B90`，最小者当选；
4. 并列时取 seed 方差更小者；
5. LoRA 组（4.4 Group B）与 AE 组用**同一规则**各选一个 checkpoint；
6. 不允许把 A90-best 与 B90-best 两个独立 checkpoint 拼成一个"统一策略"的成功率。

## 7. 闭环协议

- 16 条冻结 val episode（A90 10 条、B90 6 条），每条约 3 个推理种子（seed-offset 0/1/2）；
- 同一 episode × seed 配对表用于 AE 与 LoRA；
- execute_k=8、45 tick、12 physics steps/tick、continuous 夹爪、v1.2 严格判据；
- 不使用专家动作、不做 teacher forcing；
- 记录失败原因、推进距离、横向漂移、倾角、墙接触步数与力。
