# 实验 1 评测汇总

判据：工件抬升 ≥ 3 cm、放到目标 2 cm 内、无滑落（≤1.5 cm）、静置时关节速度收敛且已松爪。每个 episode 最长 260 个控制周期（13 s 仿真）。

## 1. 各组结果

| 组别 | 集数 | 成功 | 成功率 | 失败类型 | 平均抬升/m | 平均放置误差/m | 平均 tick |
|---|---|---|---|---|---|---|---|
| VLA 分布内 chunk50 | 20 | 13 | 65% | object_not_lifted×6, hand_contact×1 | 0.079 | 0.063 | 130 |
| VLA 分布内 n10 | 20 | 15 | 75% | object_not_lifted×4, object_slipped×1 | 0.081 | 0.045 | 129 |
| VLA 留出 chunk50 | 20 | 15 | 75% | object_not_lifted×3, object_slipped×1, hand_contact×1 | 0.089 | 0.033 | 131 |
| VLA 留出 n10 | 20 | 10 | 50% | object_not_lifted×8, object_slipped×2 | 0.062 | 0.082 | 133 |
| 专家 留出 | 20 | 20 | 100% | - | 0.100 | 0.003 | 176 |

## 2. 同种子交叉表（留出集，20 个种子）

| chunk50 \ n10 | n10 成功 | n10 失败 |
|---|---|---|
| chunk50 成功 | 8 | 7 |
| chunk50 失败 | 2 | 3 |

## 3. 留出集 n10 逐 episode 明细

| seed | 结果 | 失败类型 | 失败阶段 | 抬升/m | 放置误差/m | tick | 墙钟/s |
|---|---|---|---|---|---|---|---|
| 30000000 | 成功 | - | completed | 0.100 | 0.008 | 122 | 42.8 |
| 30000001 | 失败 | object_not_lifted | approach/grasp | 0.008 | 0.177 | 147 | 45.9 |
| 30000002 | 成功 | - | completed | 0.105 | 0.004 | 117 | 36.9 |
| 30000003 | 失败 | object_not_lifted | approach/grasp | 0.006 | 0.210 | 157 | 49.2 |
| 30000004 | 成功 | - | completed | 0.101 | 0.008 | 109 | 34.7 |
| 30000005 | 成功 | - | completed | 0.101 | 0.003 | 118 | 35.9 |
| 30000006 | 成功 | - | completed | 0.101 | 0.002 | 109 | 35.6 |
| 30000007 | 成功 | - | completed | 0.099 | 0.005 | 136 | 40.1 |
| 30000008 | 失败 | object_not_lifted | approach/grasp | 0.009 | 0.170 | 127 | 35.6 |
| 30000009 | 失败 | object_not_lifted | approach/grasp | 0.008 | 0.206 | 136 | 38.3 |
| 30000010 | 失败 | object_not_lifted | approach/grasp | 0.009 | 0.216 | 148 | 41.2 |
| 30000011 | 成功 | - | completed | 0.102 | 0.008 | 119 | 32.2 |
| 30000012 | 失败 | object_not_lifted | approach/grasp | 0.009 | 0.197 | 260 | 71.8 |
| 30000013 | 失败 | object_not_lifted | approach/grasp | 0.013 | 0.187 | 137 | 38.4 |
| 30000014 | 失败 | object_slipped | lift/transport | 0.109 | 0.021 | 118 | 32.5 |
| 30000015 | 失败 | object_slipped | lift/transport | 0.083 | 0.002 | 108 | 28.9 |
| 30000016 | 失败 | object_not_lifted | approach/grasp | 0.003 | 0.204 | 152 | 42.1 |
| 30000017 | 成功 | - | completed | 0.090 | 0.004 | 115 | 32.3 |
| 30000018 | 成功 | - | completed | 0.090 | 0.001 | 114 | 31.7 |
| 30000019 | 成功 | - | completed | 0.097 | 0.005 | 114 | 32.9 |

## 4. 推理延迟（A40，batch=1，n_action_steps=10）

| 指标 | 数值 |
|---|---|
| 新 chunk 前向 mean | 288.0 ms |
| 新 chunk 前向 p50 | 278.8 ms |
| 新 chunk 前向 p95 | 354.5 ms |
| chunk 内取动作 mean | 3.51 ms |

按 n10 摊到每个控制周期，平均 ≈ 32.3 ms，低于 20 Hz 的 50 ms 预算；单次前向 288 ms 无法直接 20 Hz 同步执行。

## 5. 视频证据（results/experiment_1/videos）

- `episode_30000000_ok.mp4`（33 KB）
- `episode_30000012_fail_object_not_lifted.mp4`（61 KB）
- `episode_30000014_fail_object_slipped.mp4`（34 KB）
