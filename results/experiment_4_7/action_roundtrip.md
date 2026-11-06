# Gate 2：Anchor-16 全行重建与裁剪审计（`action_roundtrip.csv`、`adapter_clipping_audit.json`）

## 方法

对同样 36 帧：
1. 用训练代码生成 16 步 anchor 标签（`a_k = A[t+k] − S[t]`，夹爪 `A[t+k,7]/0.04`）；
2. 经训练 Normalize（已在训练路径内）；
3. 经**部署端** `Policy._output_transform`（DroidOutputs 反归一化）→ 8D 动作；
4. 喂入正式 `ActionAdapter`（`anchor_reference='fixed'`，continuous 夹爪）；
5. 比较 `q_target[k]` 与原始专家 `A[t+k, :7]`，以及 `gripper_target[k]` 与 `A[t+k, 7]`。

## 结果

| k 区域 | 样本 | max abs（关节） | mean abs | 夹爪 max abs | accepted 比例 |
| --- | ---: | ---: | ---: | ---: | ---: |
| **k=0** | 36 | **1.1e-16** | 8.0e-17 | 6.9e-18 | 100% |
| **k=1–7** | 252 | **1.1e-16** | 7.2e-17 | 6.9e-18 | 100% |
| **k=8–15** | 288 | **1.30e-2** | 4.5e-5 | 6.9e-18 | **99.65%**（1 行被拒） |

- k=0 与 k=1–7 是**浮点精确重建**（1e-16 量级 = float64 往返误差）；
- k=8–15 的最大 0.013 rad 来自 `max_delta_rad × (k+1)` 的 fixed-anchor 裁剪（安全约束，**有意为之**），
  不是数值错误；被拒的 1 行同样是 ±1 rad 绝对偏移规则触发；
- 夹爪命令**逐位精确**（6.9e-18），continuous 模式与 `×0.04 m` 换算一致。

## 审计项（`adapter_clipping_audit.json`）

| 项 | 结果 |
| --- | --- |
| 第 k 行是否相对同一 `q_anchor` | 是（fixed 模式，`q_target[k]=q_anchor+a_k`） |
| `max_delta_rad×(k+1)` 是否生效 | 是（k8–15 的 0.013 rad 差异来自它，比例 0.35%） |
| ±1 rad 绝对偏移拒绝是否误伤 | 36×16 行中仅 1 行被拒（0.17%），且是真实大偏移 |
| 关节限位裕度是否改变目标 | 未见（accepted 行 target 与专家误差 1e-16） |
| `limit_joint_step` 是否改命令 | 专家轨迹上 **0/707 次触发**（Gate 3 trace `rate_limited=False`） |
| gripper continuous/binary | continuous（与 4.4/4.5/4.6 一致） |
| action gain | 1.0（未使用 `--action-gain`） |

## 结论

**PASS**。Anchor-16 标签经部署端反归一化与 Adapter 后，在**未触发安全约束时逐位精确**还原
全部 16 步专家目标；触发裁剪的样本比例与位置已单独记录，不与数值还原误差混算。
`execute_k=8` 只影响执行多少行，不影响训练 horizon（16）。
