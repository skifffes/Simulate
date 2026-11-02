# 实验 4.1：环境、时间基准与动作契约（Gate 1）

本文件是批量采集前的**动作/时间语义冻结记录**。所有数字都能在仓库里核到代码或数据。

- Simulate commit（采集时）：见 `artifact_manifest.json` 的 `simulate_commit`
- OpenPI commit：`215abfb217dbac7d5f1273282331b9b1866c0479`
- checkpoint：`pi05_droid`（`gs://openpi-assets/checkpoints/pi05_droid`，quantile 归一化）
- 相机：`exp4.0-camera-v3`，SHA256 `418bad2f33ef5a9e`（`src/exp40/camera_v2.py`）
- 动作契约版本：`exp4.1-action-v1`（`src/exp41/run41.py`）

## 1. 物理与策略时序

| 项 | 值 | 依据 |
| --- | --- | --- |
| 物理频率 | **240 Hz**（dt = 1/240 s） | `src/simulation.py` 的 `setTimeStep` |
| 策略频率 | **20 Hz** | `src/vla/atomic_tasks/config2.py:8-10` |
| 每个控制周期的物理步 | **12** | `240 / 20` |
| 每帧时长 | 50 ms | — |
| 一个 episode 的帧数 | A90 约 170–200，B90 约 140–180 | 采集数据实际值 |

**为什么保留 20 Hz 而不是改 15 Hz**：
1. 现有 A90/B90 专家与 `RecordingExpert` 的录制/收敛阈值都是按 12 物理步标定的（实验 2 起），
   改成 16 步会改变每个控制周期内的接触动力学与末端跟踪，从而改变专家可行性与代价；
2. 15 Hz 需要重新标定专家，属于实验 4.2 之前的工作；本次时间预算内不冒险改动已验证的采集链路；
3. **时间语义仍是一致的**：π₀.₅ 的动作是"每个控制步的关节增量"，
   只要推理侧显式声明控制步时长 = 50 ms（本仓库 `execute_k` × 12 物理步），
   训练数据的时间尺度就与推理一致；4.2 若坚持 15 Hz，应对 Δq 做时间归一化（×15/20）或重采样，
   这属于 4.2 的决定，见报告 §RQ7 / §RQ10。

## 2. 采集端的原始标签（物理动作）

`RecordingExpert`（`src/vla/atomic_tasks/recording.py`）在每个控制周期开始时记录：

```
state_t  = [q1..q7 (rad), finger1 (m), finger2 (m)]        # 9 维，实际状态
action_t = [q_target1..q7 (rad), gripper_target (m)]        # 8 维，本周期**下发的绝对关节目标**
```

然后在同一个周期内把这 8 维命令保持 12 个物理步。**所以 action 是"绝对目标"，不是增量、不是速度。**

## 3. π₀.₅（pi05_droid）要求的接口

来自 openpi 源码与 4.0 的接口审计（`src/exp40/droid_contract.py`）：

```
state(8)  = [q1..q7 (rad), gripper_normalized ∈ [0,1]]      # 0=闭合, 1=全开
action(8) = [Δq1..Δq7 (rad/控制步), gripper_normalized]     # 前 7 维是相对当前 state 的增量
归一化    = quantile（(x-q01)/(q99-q01)*2-1），模型内部宽度 32，只有前 8 维有物理含义
```

**可逆转换（本仓库实现）**：

```
Δq_t      = action_t[:7] - state_t[:7]                 # 绝对目标 → 增量（数据导出）
g_t       = clip(mean(state_t[7:9]) / 0.04, 0, 1)      # 单指位移 → 归一化夹爪
q_target  = q_now + clip(Δq, -1, 1)                    # 推理侧反变换（4.0 已实现）
width     = 0.04 * g   （预训练策略接口按 0.5 二值化，见 4.0 报告 §11）
```

- 单位：关节 rad、夹爪 m（归一化前）/ 无量纲（归一化后）；
- **0.2 系数**：属于 DROID 实机控制栈，不在 openpi 的 DROID 数据链里，本仓库**不使用**（4.0 已用测试排除重复缩放）；
- 归一化位置：训练时由 openpi transform 在 8 维上做 quantile；本次采集**不预归一化**，
  只保存物理量，并提供 `dataset_statistics.json` 的 q01/q99 供 4.2 决定。

## 4. 相位检查（自动测试）

`scripts/exp41/verify_dataset.py` 对每个 episode 计算 `|action[t][:7] - state[t+1][:7]|`：
若控制在 12 个物理步内收敛，该值应接近 0（只剩跟踪误差）；若存在**一帧相位偏移**，
该值会接近"一步的关节变化量"（0.1–0.5 rad）而 `|action[t] - state[t]|` 反而更大。

实测（全部 episode 的 `action_vs_next_state_max_rad` / `mean_rad`）见
`dataset/readback_verification.json`；数值与专家跟踪误差同量级，说明相位正确、没有错一帧。

## 5. 相机与观测

| 字段 | 内容 |
| --- | --- |
| `observation.images.top` | 外部第三人称（v3：target `(0.56,0.13,0.06)`、distance 0.70、yaw 50°、pitch −35°、fov 60°） |
| `observation.images.wrist` | 腕部（v3：link 8 固定外参 `offset (0.055,0,-0.015)`、前视 +10° 俯、fov 78°） |
| 分辨率 | 256×256×3 uint8（写入数据集）；推理时由适配器 `resize_with_pad` 到 224×224 |
| 相机版本/SHA | `exp4.0-camera-v3` / `418bad2f33ef5a9e`（随每条记录与 `meta` 一起保存） |

## 6. 数据里保存的东西（Policy Observation vs Privileged Evaluation State）

| 类别 | 字段 | 用途 |
| --- | --- | --- |
| Policy Observation | 两路 RGB、9 维 state、语言指令 | 训练/推理输入 |
| Privileged Evaluation State | 物体世界 SE(3)、速度、接触（body/link/法向/力/穿透）、EE SE(3)、EE–物体相对位姿、目标 ID、评价结果 | **仅评价/状态连接/科学分析**，不得作为策略默认输入 |

模型训练导出时使用 8 维 state（7 关节 + 归一化夹爪）与 8 维增量动作；9 维原始状态与全部审计量保留为研究工件。
