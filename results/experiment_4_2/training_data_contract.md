# 实验 4.2 训练数据契约（Gate 2）

本文件回答 prompt §3.2/§3.3/§3.4 要求的全部动作语义问题，并给出源码级依据。

- 训练导出：`exp4.1-training-export-v1`（`src/exp41/training_export.py`）
- 动作契约：`exp4.1-action-v1`（20 Hz、12 物理步/控制周期）
- 复用的官方组件：`droid_policy.DroidInputs/DroidOutputs`、`Normalize(use_quantiles=True)`、
  `TokenizePrompt`、`PadStatesAndActions(32)`、`pi0.compute_loss`、官方 optimizer/lr schedule

## 1. 完整链路（逐项核对）

```
Expert raw command                # RecordingExpert 每个控制周期下发：绝对关节目标 q_target(7) + 单指宽度 target(m)
  ↓ 记录
4.1 exported action (8D)          # action8[:7] = q_target - q_observed  (rad / control-step, 20 Hz)
                                  # action8[7]  = clip(single_finger_target / 0.04, 0, 1)   ← 用**动作目标**
  ↓ DroidInputs
OpenPI training dict              # observation/joint_position(7) + observation/gripper_position(1) → state(8)
                                  # actions(16,8)
  ↓ Normalize(quantile, DROID stats)
Normalized tensors                # state(32) / actions(16,32)（PadStatesAndActions 零填充）
  ↓ pi0.compute_loss (flow matching)
Flow-matching target u_t = noise - actions
  ↓ model prediction v_t
  ↓ Unnormalize(quantile)
8D physical action                # DroidOutputs: actions[..., :8]
  ↓ Simulate Action Adapter
Panda command                     # q_target = q_now + clip(Δq, ±1)；gripper = (g > 0.5) ? open : close
```

## 2. 逐条回答

### 2.1 模型预训练的动作是关节速度还是增量？

**是"每个控制步的关节位置增量"（rad/control-step），不是 rad/s 的速度。**
依据：DROID 的 `LeRobotDROIDDataConfig` 注释明确假设"joint velocity actions"并且**不再**应用
`DeltaActions`；而 DROID 实际数据里 `actions` 就是每步增量（openpi 的 `droid_policy.DroidOutputs`
直接把它交给控制端累加）。因此 DROID 的 q01/q99（如第 1 维 ±0.45 rad/step）是"每步增量"的量级，
不是速度量级。

### 2.2 官方是否在采集或控制端做过尺度转换？

没有额外乘性系数。DROID 的动作在采集端已经是"下一步要走的增量"，控制端按步累加。
本仓库的 4.1 专家同样记录"下一步绝对目标"，导出时**减去当前 q** 得到同样的增量语义。

### 2.3 20 Hz 数据的时间语义

我们的 Δq 是 **20 Hz 下每一步的增量**（每步 50 ms，12 个 240 Hz 物理步）。
DROID 是 15 Hz（每步 66.7 ms）。两者的**单位一致（rad/step）但每步时长不同**；
因此**不能**把 20 Hz 的 Δq 直接当作 15 Hz 的 Δq 使用（那样会慢 25%），
也**不能**用 ×15/20 反着缩放（正确换算是保持关节速度：Δq₁₅ = Δq₂₀ × 20/15 = 1.333×）。

本轮选择：**训练与推理都用 20 Hz**，不重采样、不缩放。这样 Δq 的物理含义在训练与部署两侧完全一致。

### 2.4 是否重复做了 delta 转换？

**没有。** 我们的导出已经是 delta，因此训练配置里**不使用** `DeltaActions` 变换
（与官方 droid 配置一致：`DroidInputs` 里没有 delta transform）。若再加一次 delta，
会把"增量"变成"增量的增量"，这是最容易犯的错误，已用 `training_export.audit_gripper_semantics`
和本文件的链路核对排除。

### 2.5 是否存在错误的 0.2 系数？

**不存在。** 0.2 是 DROID 实机控制栈里的缩放，不在 openpi 的 DROID 数据链里；
本仓库的 action adapter 直接用 `q_now + Δq`（rad），没有 0.2、没有 ×scale。

### 2.6 推理端执行的动作与训练标签是否同单位？

是。训练标签 = Δq(rad/step, 20 Hz) + 归一化夹爪；推理端 `Pi05PolicyClient` 收到 (16,8) →
反归一化（同一 DROID stats）→ `clip(Δq, ±1)` → `q_target = q_now + Δq`，夹爪按 0.5 二值化后
映射为 0.04 m 或 0.017 m 的指宽目标。两侧同单位、同频率。

### 2.7 夹爪：观测 vs 动作

| | 来源 | 范围（本数据） | 用途 |
| --- | --- | --- | --- |
| observation gripper | `mean(state[7:9]) / 0.04` | 0.35–1.0（实测 1.0 占多数，抓取时降到 0.35–0.5） | state 第 8 维 |
| action gripper | `action[7] / 0.04` | **0.425（闭合）/ 1.0（张开）** | 训练动作第 8 维 |

4.1 的 preflight 已证明：45/45 条 A90 都存在"观测仍张开(1.0)、动作已命令闭合(0.425)"的帧；
若用观测代替动作，抓取阶段会得到"保持张开"的错误监督。

**连续 vs 二值协议**：训练用**连续**目标（0.425 / 1.0）；部署端按官方 client 的 0.5 阈值二值化
→ 0.425 < 0.5 ⇒ 闭合，1.0 ⇒ 张开，与专家物理抓取一致（专家闭合目标 0.017 m 两指，单指 0.017 m）。
两者语义一致，且**不在同一批数据里混用**。

物理验证：4.1 的 A90 示范是真实双指夹持（finger9/10 同时接触 932–947 个物理步），
所以"闭合=0.425"确实对应能夹住方块的指宽，而不是空夹。

### 2.8 32D 动作、padding 与 mask

| 问题 | 答案（源码依据） |
| --- | --- |
| 8D 如何进入 32D | `PadStatesAndActions(32)`：state 与 actions 在最后一维补零到 32 |
| 未使用维度是否有 mask | **没有 mask**。`pi0.compute_loss` 对全部 32 维求 MSE 平均，padding 维以 0 为目标参与训练。这正是官方 `pi05_droid` 的训练方式（`action_dim=32`，只有前 8 维有物理含义） |
| action horizon | `action_horizon=16`：每个样本取 t..t+15 共 16 帧的增量，构成 (16,32) |
| episode 尾部不足 16 帧 | 采样器只在 `t ≤ T-16` 取窗口（`range(0, T-16+1, 4)`），因此**不会**在尾部产生跨 episode 的 chunk；真实执行到末尾时策略继续按同一协议预测，不训练尾部 padding |
| action chunk 是否跨 episode | **不跨**。窗口采样以 episode 为边界 |
| 模型输出如何恢复 8D | `DroidOutputs`: `actions[..., :8]`；推理端只前 8 维参与控制 |

### 2.9 频率与 chunk 执行时长

训练与推理均为 20 Hz；每次推理返回 16 步 chunk，闭环按 `execute_k`（4.0 默认 8）执行，
每个控制步 12 个物理步 = 50 ms。chunk 的真实执行时长 = `execute_k × 50 ms`（8 步 = 400 ms），
由闭环记录逐 tick 校验（不是"看起来对"）。

## 3. Gate 1 实测输出（真实 batch）

见 `dataset_batch_validation.json`：batch=4 时
`state (4,32) float64`、`actions (4,16,32) float64`、两路图像 `(4,224,224,3) float32`
（第三路 `right_wrist` 被 mask=False）、`tokenized_prompt (4,200)` mask 求和 285、
action 有效 mask 16/16、batch 内同时含 A90 与 B90。
forward loss 为有限值（`gate1_forward_loss.json`）。
