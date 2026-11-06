# Gate 4：FK / EE 坐标系审计（`fk_consistency.csv`、`first_deviation_v2.csv`）

## 问题

实验 4.6 的 `first_deviation` 报告 FK 与 rollout EE 存在约 **0.33 m 恒定偏差**，可能意味着坐标系不一致。

## 结论：**坐标定义一致；0.33 m 是 4.6 诊断脚本构造场景时的问题，不是运行时坐标系问题**

### 1. 运行时链路

`scene.sim.pose()` 的实现（`src/simulation.py:69`）：

```python
def pose(self):
    state = p.getLinkState(self.robot, self.ee, computeForwardKinematics=True, ...)
    return np.array(state[4]), np.array(state[5])
```

`self.ee` 在 B90 场景中是 **link 11 = `panda_grasptarget`**（与 Gate 4 打印的 link 表一致：
0–6 为 `panda_link1..7`，7=`panda_link8`，8=`panda_hand`，9/10 为两指，11=`panda_grasptarget`）。

### 2. 20 个合法关节构型的一致性（`fk_consistency.csv`）

| 比较 | 最大位置差 | 结论 |
| --- | ---: | --- |
| `pose()` vs 同 link（11）的 `getLinkState` | **0.0 m** | 完全一致 |
| `pose()` vs link 11 的 FK（`resetJointState`+`getLinkState(computeForwardKinematics=True)`） | **0.0 m** | 完全一致 |

→ 运行时 EE 定义、FK 链路、world 坐标、四元数约定**没有不一致**；
不存在 flange/TCP/tool offset 未被计入的问题（`panda_grasptarget` 本身即工具点）。

### 3. 4.6 的 0.33 m 从哪来

4.6 的脚本在**纯 CPU 侧新建**了一个 B90 场景来做 FK，用 `robot_q=`/`obj_pos=` 复现初始条件；
而 rollout 的 EE 来自**实际执行场景**。两者在"专家 EE 参考"上相差约 0.33 m 且为常数，
说明是**诊断脚本里的场景/初始状态构造**（例如机器人 base 或物体位姿复现）与运行时不一致，
而不是 `pose()`/FK 的坐标系错误。

**验证**：改用**同一运行时路径**重算（见 §4），tick0 的 EE 距离为 **0.0 m**（精确到 1e-5 m），
这直接排除了坐标系不一致的假设。

### 4. 重算 first deviation（`first_deviation_v2.csv`）

方法：
- 专家 EE = Gate 3 **模式 A 专家回放** `expert_execution_trace.csv` 中每 tick 的 `ee_before`（运行时 `pose()`）；
- rollout EE = 实验 4.6 step3000 的 `*_substeps.csv` 每 tick 第一个子步的 `ee_before`（同一 `pose()`）。

| 指标 | 结果 |
| --- | --- |
| tick0 EE 距离 | **0.0 m**（36/36 条；个别 1e-5 m） |
| 首次 >3 cm 且持续（≥80% 后续 tick >1 cm）的 tick | **1–2**（B90 多数为 1，A90 为 1–2） |
| 45 tick 内最大 EE 偏离 | 0.33–0.68 m |

**注意措辞**：这只是"首次超过 3 cm 阈值且持续的 tick"，**不是"首次不可恢复状态"**——
本轮没有定义可恢复性判据，因此不声称找到了不可恢复点。

## 对后续的意义

- 4.6 的 0.33 m **不是契约 bug**，只是诊断脚本的参考构造问题；用运行时 EE 即可对齐；
- 真正可报告的结论是：step3000 模型从专家初态出发，**第 1–2 个控制周期就偏离专家 >3 cm**，
  且此后再未回到专家轨迹 1 cm 以内——这与"模型动作与专家标签的差异"直接相关（见误差预算）。
