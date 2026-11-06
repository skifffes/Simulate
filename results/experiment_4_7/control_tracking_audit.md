# Gate 3：20 Hz 物理执行一致性（`expert_execution_trace.csv`、`oracle_chunk_results.csv`）

四层结构（逐 tick 记录在 trace）：**Expert Target → Adapter Target → Controller Command → Actual Motion**。

- 模式 **A**：每 tick 直接执行专家 `A[t]`（不经过 chunk）；
- 模式 **B1**：chunk 起点用当前真实 q 重锚定**专家绝对目标**，Adapter 后执行前 8 行；
- 模式 **B2**：chunk 起点用**原始训练标签**（`A[t+k] − S_recorded[t]`）在当前 q 上重锚定，执行前 8 行；
- **safe=False**：仅模式 A 的隔离消融（关闭 `limit_joint_step` 与分类安全约束），用于对照。

## 严格成功（v1.2，未改阈值）

| 模型/模式 | A90 ep0 | A90 ep5 | B90 ep0 | B90 ep2 | 合计 |
| --- | --- | --- | --- | --- | ---: |
| **A（专家逐周期）** | ✅ | ✅ | ✅ | ✅ | **4/4** |
| **B1（Oracle 绝对目标 chunk）** | ✅ | ✅ | ✅ | ✅ | **4/4** |
| B2（原始标签重锚定） | ✅ | ✅ | ✅ | ✗ `wall_contact_ever` | 3/4 |
| A（safe=False 消融） | ✅ | ✅ | ✅ | ✅ | 4/4 |

## 跟踪与控制统计（707 个 tick/模式）

| 模式 | 关节跟踪 max | 跟踪 mean | Adapter 目标 vs 专家 max | rate-limited | 接受率 | 夹爪跟踪 max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A（safe） | **0.0121 rad** | 0.0017 | **0.0** | 0/707 | 707/707 | 0.0065 |
| A（unsafe） | 0.0121 rad | 0.0017 | 0.0 | 0/707 | 707/707 | 0.0065 |
| **B1** | **0.0121 rad** | 0.0017 | **0.0** | 0/707 | 707/707 | 0.0065 |
| B2 | 0.1506 rad | 0.0036 | **0.4377 rad** | 0/707 | 707/707 | 0.0116 |

## 判读

1. **专家逐周期回放 4/4 严格成功** → 数据、场景、控制参数、夹爪单位、20 Hz/240 Hz 时序在物理上可完整复现；
2. **B1 Oracle chunk 也 4/4 成功，且 Adapter 目标与专家逐位相同（0.0）** →
   chunk 语义、fixed anchor、Adapter、8 步执行、重新锚定链条**没有引入误差**；
3. **B2 的差异（0.44 rad）来自"用录制状态作锚点"的语义**，不是代码 bug：
   当实际状态偏离录制状态后，`A[t+k]−S_recorded[t]` 会被整体平移；B2 的失败（B90 ep2）是这种平移导致的，
   与 4.6 观察到的"闭环状态漂移"一致；
4. **安全约束在本轮专家轨迹上 0 次触发**（`rate_limited=0/707`、接受率 100%），
   因此"限速/裁剪是否使机器人跟不上专家"在本数据上**没有证据**；
   unsafe 消融与 safe 成功率相同（4/4）；这不能外推为"安全约束永远无影响"（模型轨迹另算，见 Gate 5/4.6）。

## 源码位置

- Adapter：`src/exp40/action_adapter.py::chunk_to_targets` / `limit_joint_step`；
- 执行：`scripts/exp47/gate3_replay.py`（12 physics steps/tick，POSITION_CONTROL，force=30）；
- 严格判据：`src/exp41/strict_eval.py`（`exp4.1-strict-v1.2`，阈值未改）。
