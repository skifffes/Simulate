# 官方语义审计（Hotfix B / C 的证据链）

对象：`pi05_droid`（`gs://openpi-assets/checkpoints/pi05_droid`），openpi commit
`215abfb217dbac7d5f1273282331b9b1866c0479`。所有引用都是 openpi 仓库内路径 + 行号。

## 1. 夹爪：`0/1` 的物理含义

### 1.1 pi 内部夹爪空间（0 = closed，1 = open）

`src/openpi/policies/aloha_policy.py:117-137`：

```python
def _gripper_to_angular(value):
    # ...
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = _unnormalize(value, min_val=0.01844, max_val=0.05800)
```

`constants.py:14-15`：`PUPPET_GRIPPER_POSITION_OPEN = 0.05800`、`PUPPET_GRIPPER_POSITION_CLOSE = 0.01844`。

因此 **pi 内部归一化夹爪 0 → CLOSE、1 → OPEN**（`_unnormalize` 用 min=CLOSE、max=OPEN 线性映射）。

反向 `aloha_policy.py:140-150`（`_gripper_from_angular`，把 pi 空间动作映射回 Aloha）：
`value + 0.5476` 后再用 `min=-0.6213`（PUPPET_GRIPPER_JOINT_CLOSE）、`max=1.4910`（OPEN）归一化——
同样是 **pi 输出越大越开**。

### 1.2 DROID checkpoint 的夹爪统计与官方客户端用法

- checkpoint 自带 `assets/droid/norm_stats.json`：gripper 维（第 8 维）`q01=0.0`、`q99=0.9998`、`mean=0.4514`
  → 是**绝对值**、范围 [0,1]，不是随 state 变化的增量。
- 官方 DROID 客户端 `examples/droid/main.py:141-146`：`action[-1] > 0.5` 时置 1，否则置 0；
  `:80` `RobotEnv(action_space="joint_velocity", gripper_action_space="position")`。

### 1.3 结论（本仓库实现）

| 环节 | 本仓库实现 | 是否与官方语义一致 |
| --- | --- | --- |
| 观测 | `g = mean(finger1,finger2)/0.04`，夹到 [1e-4, 0.9999] | 一致（1=张开时 g→1） |
| 动作 | `g>0.5 → 1 → 单指目标 0.04 m`；`g≤0.5 → 0 → 0.0 m` | 一致（1=开） |
| 一致性 | 观测与动作用同一方向 | 一致 |

**保留原实现**（4.0 的方向本来就是对的），本次只补了可反驳证据：
`hotfix_semantics_tests.json` B 组 7/7，其中真实仿真里张开 `open_m=0.0400`、闭合 `closed_m=0.0003`，
观测标量 `g_open=0.9999` / `g_closed=0.0080`。

**未关闭的风险**：官方 DROID 数据本身（`droid` 数据集）的 gripper 编码方向只能从 pi 内部空间反推，
没有 DROID 实机文档逐条核对；4.1 若采集自己的数据，应做一次“开/合”显式对照。

## 2. 七关节动作的尺度

### 2.1 模型输出已经是物理量

`src/openpi/policies/policy_config.py:73-84` 的输出链是
`model_transforms.outputs → Unnormalize → data_transforms.outputs(DroidOutputs)`；
`transforms.py:175-181` 的 quantile 反归一化 `(x+1)/2*(q99-q01+1e-6)+q01` **只在 8 维内**生效，
8 维之后原样透传（`PadStatesAndActions(32)` 的填充维因此没有物理含义）。

反归一化后即物理单位（rad/步），checkpoint 统计给出区间：
`[-0.458,0.448] / [-0.808,0.765] / [-0.447,0.448] / [-0.927,0.794] / [-0.646,0.648] / [-0.646,0.663] / [-0.762,0.734]`，
7 维都以 0 为中心（相对不对称度 < 0.30，见测试 C），这就是**关节增量**而不是绝对关节角
（绝对角会像 state 一样有 ±2.8 rad 的量级和明显 nonzero mean）。

### 2.2 关于 `0.2 × joint_velocity`

- 该系数属于 **DROID 实机控制栈**（`droid` 包的 RobotEnv：动作 [-1,1] × 每步最大关节速度），
  **不在 openpi 的 DROID 训练/推理数据链里**：`droid_policy.py` 直接返回 `actions[..., :8]`，
  `config.py:403-409`（RLDS JOINT_POSITION 分支）只做 `DeltaActions`（减去 state），没有乘 0.2。
- 因此在本仓库里**不应再乘 0.2**：那会把“关节增量”二次缩小。
- 本仓库的实现：`q_target = q_current + clip(a, -1, 1)`，适配器不引入任何缩放；
  超过 ±1 的动作按官方 `examples/droid/main.py:149` 的 clip 语义**拒绝并记录**（不静默缩放）。

证据：`hotfix_semantics_tests.json` C 组 5/5，其中
`q_target - q == action[:7]`（逐位相等）、`≠ 0.2 × action[:7]`。

## 3. 控制频率

- DROID 采集/执行 15 Hz：`examples/droid/main.py:23` `DROID_CONTROL_FREQUENCY = 15`；
  `:42` `open_loop_horizon = 8`。
- 15 步 chunk 在 15 Hz 下 = **1.00 s**；每 tick 16 个 240 Hz 物理步（`hotfix_semantics_tests.json` D 组）。

## 4. 视觉约定

- 输入键名与第三路相机占位：`src/openpi/policies/droid_policy.py:55-61`（`right_wrist_0_rgb` 用
  `np.zeros_like(base_image)` + `image_mask=False`）。
- 官方客户端在机器人侧 `resize_with_pad → 224×224`（`examples/droid/main.py:120-123`），
  服务端 `ResizeImages(224,224)` 因此是 no-op。
- **本次明确不声称**：Simulate 的相机外参与 DROID 训练分布对齐。只保证几何可见性、固定可复现、格式正确。
