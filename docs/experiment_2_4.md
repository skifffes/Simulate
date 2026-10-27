# 实验 2.4：VLA 原子技能组合中的接触模式失效归因与多视角诊断

## 摘要

本轮不训练模型、不修改任务判据，只做**可复现的接触诊断**。用冻结的 D2 checkpoint（实验 2.3 的 diverse-start 模型），在 PyBullet 中跑了 6 组正式对照、每组 10 个 episode，共 **60 次 rollout**；每次 rollout 记录 240 Hz 物理状态、逐次策略动作、逐物理步接触点，以及 **top / side / contact 三视角视频**。共产出 **180 个基础视角视频、60 个三视角 comparison、46 个事件剪辑**（14 个 rollout 没有检测到可剪辑事件）。

| 组别 | 条件 | 成功 | 抬升>3cm | 夹爪双指接触 | 桌面支撑丢失 | 墙接触 | 最大抬升均值 |
|---|---|---:|---:|---:|---:|---:|---:|
| G0 | A0→B 直接 | **0/10** | 7/10 | 8/10 | 7/10 | **0/10** | 6.8 cm |
| G1 | A90→B 直接 | **0/10** | 10/10 | 10/10 | 10/10 | **0/10** | 9.3 cm |
| G2 | B 独立 HOME yaw0 | **0/10** | 2/10 | 8/10 | 3/10 | 8/10 | 1.8 cm |
| G3 | B 独立 HOME yaw90 | **9/10** | 0/10 | 10/10 | 1/10 | 0/10 | 0.6 cm |
| G4 | A0→B 连续退回 HOME | **0/10** | 4/10 | 7/10 | 7/10 | 8/10 | 3.4 cm |
| G5 | A0→B 局部短退让 | **0/10** | 0/10 | 0/10 | 0/10 | 0/10 | 0.04 cm |

三个核心结论：

1. **A0→B / A90→B 的失败不是"推不动"，而是 B 在交接后立刻闭合夹爪、把物体夹住并抬起来**。G0/G1 的物体最大抬升均值分别 6.8 / 9.3 cm，双指同时接触 8/10 和 10/10，桌面支撑丢失 7/10 和 10/10；**两组都没有任何一次物体—墙壁接触**。这直接支持 H1（模型主动向上）与 H2（形成抓持），基本排除 H3（墙挤压抬升）在 G0/G1 中的作用。
2. **同一个 B 在独立 yaw90 起点上是干净的推动**（G3 9/10，最大抬升 0.6 cm），说明模型没有"不会推"；差异来自**交接状态下的夹爪-物体相对位姿**。A 的终态让夹爪已经悬在物体上方/两侧，B 的第一批动作就是闭合与抬升；独立起点则从 HOME 侧向接近，先形成推动接触。
3. **yaw0 是另一条独立的失败链**：G2（B HOME yaw0）0/10，但失败形态不同——8/10 撞墙、4/10 翻倒、4/10 推转、最大抬升只有 1.8 cm。推 4 cm 短端面时物体绕 Z 旋转/翻倒，而不是被抓起来。A0 的终态正好是 yaw0，所以 G4（真实退回 HOME）仍然 0/10：**退让恢复了机器人构型，但没有改变物体朝向**。

因此，A→B 的失败至少有两层：**交接构型导致的"抓+搬"** 与 **yaw0 的接触不稳定**；它们不能用一个"回 HOME"或"扩大起点"同时解决。

---

## 1. 实验设置

| 项 | 值 |
|---|---|
| checkpoint | `outputs/smolvla_e23_D2/checkpoints/020000/pretrained_model`（sha256 `34a6e308…`） |
| 代码 | 本轮实现 `scripts/run_experiment_2_4.py`，基于仓库 `fcf7665` 之后的工作区 |
| 仿真 | PyBullet DIRECT，240 Hz 物理，20 Hz 策略，每动作保持 12 个物理步 |
| 观测 | 与训练完全一致：top + wrist 256×256 RGB，9 维本体状态；**诊断相机不作为 VLA 输入** |
| 动作 | 7 关节位置 + 夹爪目标，8 维，chunk 由策略内部维护 |
| 测试种子 | G0–G5 均 40000000–40000009 |
| 判据 | 统一 v2.2：任务条件 → 停发动作 → 静置 1.0 s → 复核 gap ≤2 cm、\|dy\| ≤2 cm、dyaw ≤15°、tilt ≤12°、speed <0.01 |
| 视频 | 20 fps，640×480，H.264；每 episode 保存 top / side / contact / comparison / events |
| 日志 | 240 Hz 物理 CSV、接触 CSV、策略 JSONL、成功判据时间线、frame_index.csv |

新增相机：

- **top**：原固定第三人称相机（仅提高渲染分辨率，不改变 VLA 输入）。
- **side**：位于 `(0.55, −0.62, 0.30)`，看向 `(0.60, 0.16, 0.10)`，FOV 46°，能同时看到桌面、物体、夹爪和挡板（见 `side_*.png`）。
- **contact**：复用 wrist 相机的高分辨率渲染，近距离显示左右指与物体。

## 2. 统一时间基准

每个 episode 维护以下全局索引，全部写入日志与 `frame_index.csv`：

| 字段 | 定义 |
|---|---|
| `physics_step` | 从场景初始化后第一次 `sim.step()` 起全局单调递增，240 Hz |
| `physics_time_sec` | `physics_step / 240` |
| `policy_step` | 每发送一次 20 Hz 动作 +1；retreat/settle 期间每 12 个物理步 +1 |
| `frame_index` | 每渲染一帧 +1；一帧对应一个 policy tick（retreat 期间每 12 步一帧） |
| `skill` | `A0 / A90 / B / retreat / settle` |
| `switch` 事件 | 记录 `from/to/policy_step/physics_step/physics_time_sec/frame_index` |

`frame_index.csv` 的每一行把 `frame_index → physics_step → physics_time_sec → policy_step → skill` 绑定。验证示例（G0 seed 40000002）：

```
switch = {policy_step: 97, physics_step: 1194, physics_time_sec: 4.975, frame_index: 97}
frame_index.csv 第 97 行 = {physics_step: 1194, time: 4.975, policy_step: 97, skill: A0}
```

也就是说，切换前的最后一帧、切换事件与物理日志严格落在同一 `physics_step` 上；不再出现"`switch.tick=97` 却和 `ticks[97]` 对不上"的问题。每个视频用 `probe_video` 校验了实际帧数与 FPS（例如 357 帧 / 20 fps / 17.85 s），`frame_index.csv` 的帧数与视频帧数一致，没有用推测帧率对齐。

## 3. 日志与接触解析

每个 episode 目录包含：

| 文件 | 内容 |
|---|---|
| `physics_trace.csv` | 240 Hz：q/dq/q_target、夹爪实际/目标、末端位姿与速度、物体位姿/速度/倾角/墙距/抬升、接触汇总、成功判据 |
| `contact_trace.csv` | 每个真实接触点：body/link ID、link 名称、接触点世界坐标、法向、法向力、距离、摩擦 |
| `policy_trace.jsonl` | 每次策略调用：skill、task、本体状态、返回动作、延迟、对应 frame |
| `success_timeline.csv` | 每个 B 控制周期：gap、lateral、dyaw、speed、tilt 与各子条件、完整判据 |
| `frame_index.csv` | 视频帧 ↔ 物理步 ↔ 策略步 ↔ 技能 |
| `event_summary.json` | 首次接触、双指、桌面丢失、墙壁接触、抬升、最大抬升、首次成功判据等 |
| `metadata.json` | checkpoint、sha256、commit、PyBullet 版本、频率、分辨率、阈值 |
| `*.mp4` | top / side / contact / comparison / events |

接触解析按 **body id + link id** 判定，不按接触点数量：左指 = robot link 9，右指 = link 10，手掌 = link 8；桌面 = plane，墙 = wall body。`left_finger_contact` / `right_finger_contact` 是当步是否与物体接触，`left_force/right_force/wall_force` 是该类接触法向力之和。

## 4. 六组正式结果

正式测试每组 10 个 episode；冒烟测试（每组 1–2 次）已完成但**不计入**下表。

### 4.1 总表

| 组别 | n | 成功 | A 成功 | 双指接触 | 桌面丢失 | 墙接触 | 抬升>3cm | max lift 均值 / 最大 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| G0 A0→B | 10 | 0 | 10 | 8 | 7 | 0 | 7 | 6.8 / 10.6 cm |
| G1 A90→B | 10 | 0 | 10 | 10 | 10 | 0 | 10 | 9.3 / 10.4 cm |
| G2 B yaw0 | 10 | 0 | — | 8 | 3 | 8 | 2 | 1.8 / 6.3 cm |
| G3 B yaw90 | 10 | **9** | — | 10 | 1 | 0 | 0 | 0.6 / 2.1 cm |
| G4 A0→B 退 HOME | 10 | 0 | 10 | 7 | 7 | 8 | 4 | 3.4 / 10.6 cm |
| G5 A0→B 局部退让 | 10 | 0 | 10 | 0 | 0 | 0 | 0 | 0.04 / 0.1 cm |

### 4.2 失败类型

| 类型 | 数量 | 主要组 |
|---|---:|---|
| `grasp_lift`（双指抓持 + 离开桌面 + 抬升>3cm） | 18 | G0 7、G1 9、G2 1、G4 1 |
| `tipping`（物体倾斜 >20°） | 15 | G2 4、G4 5、G5 4、G0 1、G3 1 |
| `no_contact`（B 阶段从未碰物体） | 5 | G5 5 |
| `wall_induced_lift`（墙接触早于抬升，且抬升>3cm） | 4 | G2 1、G4 3 |
| `push_rotated`（撞墙但 yaw/侧移超差，无大抬升） | 4 | G2 4 |
| `not_reached`（没有墙接触且 gap>2cm） | 3 | G0 2、G5 1 |
| `success_then_lost`（达到完整判据后最终失败） | 1 | G1 1 |
| `success` | 9 | G3 9 |
| `unknown` | 1 | G4 1 |

## 5. 关键案例逐秒分析

所有路径均相对服务器 `~/simulate_vla/results/experiment_2_4/`。

### 5.1 G0 A0→B 典型失败（seed 40000002，max lift 10.6 cm）

- 视频：[top.mp4](file:///scratch/lijh/simulate_vla/results/experiment_2_4/G0_a0b_direct/seed_40000002/top.mp4)、[side.mp4](file:///scratch/lijh/simulate_vla/results/experiment_2_4/G0_a0b_direct/seed_40000002/side.mp4)、[contact.mp4](file:///scratch/lijh/simulate_vla/results/experiment_2_4/G0_a0b_direct/seed_40000002/contact.mp4)
- 事件表：

| 事件 | 物理步 | 时间 (s) | frame | 物体 x / z (m) | EE z (m) |
|---|---:|---:|---:|---|---:|
| SKILL SWITCH A0→B | 1194 | 4.975 | 97 | 0.553 / 0.020 | 0.034 |
| 首次接触 | 1195 | 4.98 | 98 | 0.553 / 0.020 | 0.034 |
| 双指同时接触 | 1203 | 5.01 | 98 | 0.553 / 0.025 | 0.043 |
| 桌面支撑丢失 | 1205 | 5.02 | 98 | 0.553 / 0.027 | 0.046 |
| 抬升 >3 cm | 1263 | 5.26 | 103 | 0.557 / 0.050 | 0.068 |
| 最大抬升 | — | — | — | 0.694 / 0.079 | — |

- 接触关系：首次接触同时出现在 link 9/10（左/右指）；抬升瞬间双指接触、`table_contact=0`、`wall_contact=0`，夹爪实际开度 0.021 m。
- 抬升前 0.2 s 内：EE 实际 z +7.9 mm，目标 z（离线 FK）+7.1 mm；物体 z 在随后 0.25 s 上升 3 cm。
- 判断：**H1 + H2 成立**。B 在切换后 5 ms 内接触、30 ms 内双指闭合、50 ms 内失去桌面支撑；机械臂目标与实际上都在向上运动，物体跟随夹爪上升。没有墙接触，H3 不成立。
- 仍不能排除：夹爪闭合是指令直接导致还是接触力把手指推闭合；需要看 `policy_trace.jsonl` 的 width 指令序列（该序列在抬升前已经从 0.035 降到 0.021）。

### 5.2 G1 A90→B 典型失败（seed 40000005，max lift 10.4 cm）

- 视频：[side.mp4](file:///scratch/lijh/simulate_vla/results/experiment_2_4/G1_a90b_direct/seed_40000005/side.mp4)
- 事件：switch 7.275 s（policy 143）；首次接触 7.28 s，仅 **link 9（左指）**；双指 7.42 s；桌面丢失 7.80 s；抬升>3cm 9.76 s；最终物体 x=0.694、z=0.057、gap=3.4 cm。
- 抬升前 0.2 s：EE 实际 +15.4 mm，目标 +27.0 mm；双指接触、桌面 0、墙 0。
- 判断：与 G0 相同的"抓+搬"模式，但建立抓持更慢（左指先接触，双指晚 0.14 s），抬升发生在切换后 2.5 s。A90 终态的腕部旋转让左右指不是同时接触，而是先单指、再夹住。
- 对照：G1 的 10 集里 10 集双指、10 集桌面丢失、0 集墙接触，说明这不是个别 seed 的偶发。

### 5.3 G2 B 独立 HOME yaw0（seed 40000000，失败但机制不同）

- 视频：[side.mp4](file:///scratch/lijh/simulate_vla/results/experiment_2_4/G2_b_home_yaw0/seed_40000000/side.mp4)
- 事件：首次接触 2.75 s（仅左指）、双指 3.05 s、抬升>3cm 4.00 s（但此刻 EE z 反而比 0.2 s 前低 2.7 mm，且桌面仍接触、只有左指接触）、墙接触 4.90 s、最终倾角 **61°**、gap 6 mm。
- 判断：这不是"抓起来"，而是**单指/边缘接触把物体推转、随后靠墙翻倒**。yaw0 的 4 cm 短端面被推时力矩不平衡，8/10 集撞墙、4/10 集翻倒、4/10 集推转；最大抬升均值只有 1.8 cm。
- 与 G0/G1 的差别：G0/G1 是双指夹持抬升，G2 是单侧接触 + 墙导致的倾倒/旋转。

### 5.4 G3 B 独立 HOME yaw90（seed 40000000，成功）

- 视频：[side.mp4](file:///scratch/lijh/simulate_vla/results/experiment_2_4/G3_b_home_yaw90/seed_40000000/side.mp4)
- 事件：首次接触 2.71 s（右指）、双指 3.08 s、成功判据 5.22 s；最大抬升 0.6 cm，最终 gap 1.66 cm、tilt 0.01°。
- 判断：长端面推动，接触稳定，物体始终贴桌；没有抬升、没有墙冲击。G3 9/10 成功，唯一失败是一次 tipping，不是抓取。

### 5.5 G4 A0→B 退回 HOME（seed 40000000）

- 视频：[side.mp4](file:///scratch/lijh/simulate_vla/results/experiment_2_4/G4_a0b_retreat_home/seed_40000000/side.mp4)
- 事件：A 在 5.075 s 成功 → 连续关节退回 HOME（约 7 s）→ B 首次接触 12.03 s（右指）、双指 12.06 s；最终 gap 1.98 cm、侧向位移 22.9 cm（相对场景初始，包含 A 的搬运）、没有成功。
- 判断：退让确实把机器人恢复了（连续关节运动，无瞬移），但**物体仍是 yaw0**，B 又回到 G2 的 yaw0 失败模式：撞墙/推转/偶尔抬升。G4 10 集墙接触 8、翻倒 5、抬升>3cm 4，和 G2 同族，而不是 G0 的抓取族。
- 这解释了实验 2.2 的 A0→B 退让只有 4/20：退让只能修复机器人构型，修不了物体朝向。

### 5.6 G5 A0→B 局部短退让（seed 40000002）

- 视频：[side.mp4](file:///scratch/lijh/simulate_vla/results/experiment_2_4/G5_a0b_local_retreat/seed_40000002/side.mp4)
- 退让：IK 到当前末端 `(-4 cm, 0, +3 cm)`，关节空间 0.002 rad/步连续执行（`local_retreat_mode=ik_offset`），无瞬移、无专家推动。
- 结果：切换 4.975 s 后 B 在 260 tick 内**从未接触物体**（无 first_contact），最终 gap 20.6 cm；G5 10 集里 5 集 no_contact、4 集 tipping、0 集成功、最大抬升 0.1 cm。
- 判断：预先固定的 5 cm 退让把夹爪退到了 B 够不到物体的位置；这个特定局部退让不成立。它不能证明"所有局部调整都无效"，只说明**退让距离必须和 B 的接近能力匹配**。

## 6. 机制判定（H1–H6）

| 假设 | 证据 | 判定 |
|---|---|---|
| H1 模型主动向上 | G0 抬升前 EE 目标 z +13.9 mm（10 集均值）、实际 +12.3 mm；G1 目标 +21.1 / 实际 +16.1 mm；物体随后才抬升 | **支持**（G0/G1） |
| H2 形成抓持 | 抬升瞬间双指接触 G0 7/7、G1 10/10；桌面接触 0；物体失去支撑后继续跟随夹爪 | **支持**（G0/G1） |
| H3 墙挤压抬升 | G0/G1 墙接触 0/10；G2/G4 有墙接触且部分抬升，但最大抬升均值仅 1.8/3.4 cm，且发生在撞墙之后 | **仅在 G2/G4 的部分 episode 支持**，不是 G0/G1 主因 |
| H4 推动接触位置不合理 | G2 yaw0 首次接触只有单指，抬升瞬间仍单指+桌面接触，随后翻倒；G3 yaw90 首次接触单指但双指后稳定推动 | **支持**（yaw0 单侧接触） |
| H5 完成后未及时停止 | G1 seed 40000001 在 policy 314 达到完整判据，但 1 s 静置后判据失败（物体最终 z=0.065、侧移 19.7 cm） | **1 例支持**；其余失败从未达到判据 |
| H6 评测/队列错误 | 切换时 `policy.reset()`；语言任务正确；动作维度/单位与 2.3 一致；A 与 B 动作块没有交叉执行；G3 同一 B 能成功 | **未发现证据** |

## 7. 研究问题回答

**Q1：交接之后夹爪和物体之间实际发生了什么接触？**
G0 在切换后 5 ms 内接触物体，30 ms 内双指接触，50 ms 内失去桌面支撑；G1 先左指接触，0.14 s 后双指，0.5 s 后桌面丢失。接触集中在指面（link 9/10），没有手掌/臂部接触。

**Q2：物体抬升时，末端是否也主动向上？**
是。G0 抬升前 0.2 s 目标 z +13.9 mm、实际 +12.3 mm；G1 目标 +21.1 mm、实际 +16.1 mm。目标上升早于物体抬升，支持"模型生成了向上动作"。

**Q3：物体是否受到左右夹爪同时夹持？**
G0 抬升时 7/7 双指；G1 10/10 双指；G2 抬升时只有左指 + 桌面仍接触（不是夹持）；G3 推成功时双指接触但不抬升。

**Q4：接近墙壁后是否发生挤压/翻转/夹带？**
G0/G1 完全没有墙接触，所以没有"墙挤压"。G2（yaw0）8/10 撞墙，4/10 翻倒、4/10 推转；G4 8/10 撞墙、5/10 翻倒、3/10 抬升发生在墙接触之后。墙是 yaw0 失败链的一部分，但不是 G0/G1 抬升的原因。

**Q5：B 是否达到过成功条件但没及时终止？**
只有 G1 seed 40000001 达到过完整判据（policy 314），但正式协议在判据满足时立即停发动作并静置 1 s；静置后物体侧移 19.7 cm、z=0.065，复核失败。这是 `success_then_lost_during_settle`，不是"继续执行把成功破坏"；其余失败从未达到判据。

**Q6：同一 B 在独立起点与 A 交接起点下为什么不同？**
G3（独立 yaw90）9/10 成功、最大抬升 0.6 cm；G1（A90 交接）0/10、最大抬升 9.3 cm。A90 终态让夹爪已经位于物体上方/两侧、腕部旋转，B 的第一步就是闭合/抬升；独立起点从 HOME 侧向接近，先形成推动接触。**状态决定动作模式，而不是模型缺少推动能力。**

**Q7：不回 HOME 是否存在简单、安全、短距离调整？**
本轮预先固定的 5 cm 局部退让（−4 cm X、+3 cm Z）**失败**：5/10 完全没有接触物体，0/10 成功。原因是退让后夹爪超出了 B 的接近范围。这个结果只否定了该固定偏移，不否定所有局部调整；更小的偏移（如 1–2 cm）需要在下一轮预先注册后测试。G4 的完整 HOME 退让也没有改善（0/10），因为物体仍是 yaw0。

**Q8：实验 2.3 的多起点模型是否改变了接触行为，还是只改变了成功率？**

在相同的 5 个 seed（40000000–40000004）上做了 5 集诊断（D0/D1/D2 各 5 次，**不是正式统计**）：

| 模型 | 成功 | grasp_lift | 双指 | 桌面丢失 | 墙接触 | max lift 均值 |
|---|---:|---:|---:|---:|---:|---:|
| D0（实验 2） | 0/5 | 5/5 | 5/5 | 5/5 | 0/5 | 6.8 cm |
| D1（canonical 重采） | 0/5 | 2/5 | 5/5 | 5/5 | 0/5 | 4.2 cm |
| D2（多起点） | 0/5 | 3/5 | 3/5 | 3/5 | 0/5 | 6.3 cm |

三者在 A→B 上都是 0/5 成功、0/5 墙接触，双指与桌面丢失都出现；D0 全部抓到 3 cm 以上，D1 有 5/5 双指但只有 2/5 抬过 3 cm（更像夹住后滑落），D2 介于两者之间。结论：**多起点训练没有改变 A0→B 的接触失效模式**（仍然是"夹住/抬起/带走"），它改变的是实验 2.3 里独立起点上的原子成功率；交接状态下的动作模式没有被数据覆盖改变。5 集样本只能作为方向性证据。

## 8. 图表

- 图 1：[fig1_object_motion.png](../results/experiment_2_4/figures/fig1_object_motion.png) — G0 最严重抬升案例的物体 X/Z 与事件线。
- 图 2：[fig2_ee_vs_object.png](../results/experiment_2_4/figures/fig2_ee_vs_object.png) — EE 实际 Z、EE 目标 Z（离线 FK）、物体 Z；目标先于物体上升。
- 图 3：[fig3_contact_forces.png](../results/experiment_2_4/figures/fig3_contact_forces.png) — 左右指、墙、总法向力。
- 图 4：[fig4_gripper.png](../results/experiment_2_4/figures/fig4_gripper.png) — 夹爪实际/目标开度与左右指接触。
- 图 6：[fig6_success_predicate.png](../results/experiment_2_4/figures/fig6_success_predicate.png) — B 各子条件与完整判据时间线。
- 图 7：[fig7_failure_distribution.png](../results/experiment_2_4/figures/fig7_failure_distribution.png) — 60 次 rollout 的失败类型分布。
- 图 8：[fig8_group_rates.png](../results/experiment_2_4/figures/fig8_group_rates.png) — 各组成功率、抬升率、墙接触率、双指接触率。

![Figure 8](../results/experiment_2_4/figures/fig8_group_rates.png)

## 9. 失败分类说明与阈值

- `grasp_lift`：双指在抬升时刻同时接触、桌面接触为 0、B 阶段最大抬升 >3 cm。
- `wall_induced_lift`：首次墙接触早于首次抬升，且最大抬升 >3 cm。
- `tipping`：物体 tilt 在任意物理步超过 20°。
- `push_rotated`：有墙接触、抬升 ≤3 cm，但最终 dyaw >15° 或 \|dy\| >2 cm。
- `not_reached`：没有墙接触且最终 gap >2 cm。
- `no_contact`：B 阶段从未出现 robot–object 接触。
- `success_then_lost`：`success_timeline.csv` 中完整判据曾为 1，但最终复核失败。
- `unknown`：不满足以上任何一条，保留不强行分类。

这些规则是**可追溯的**：每条都能在 `physics_trace.csv`、`contact_trace.csv`、`success_timeline.csv` 中找到对应字段。

## 10. 局限

- 每组 10 次，置信区间较宽；G0/G1 的 0/10 只能说明"在这 10 个种子上全部失败"，不能给出精确失败率。
- 视频在 640×480、20 fps 下录制；成功 episode 的 1 s 静置阶段没有渲染进视频（物理日志与 `success_timeline.csv` 覆盖），这是录制策略的限制，不是时间对齐错误。
- 仅有 D2 checkpoint 的正式六组；D0/D1 的接触行为只做了 5 集诊断（见 Q8），不足以做统计结论。
- 局部退让只测了一个预注册偏移；没有测试 1–2 cm 的更小偏移，也没有做碰撞/可达性搜索。
- 诊断相机不参与 VLA 输入；这一点在 `metadata.json` 中显式记录。
- 没有重新训练、没有改判据，也没有把诊断性提前停止混入正式成功率。

## 11. 复现与产物

```bash
# 单组正式运行（服务器）
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src venv/bin/python \
  scripts/run_experiment_2_4.py --group G0_a0b_direct --episodes 10 \
  --seed-base 40000000 \
  --policy outputs/smolvla_e23_D2/checkpoints/020000/pretrained_model \
  --checkpoint-sha 34a6e3080a88250302a9f0abbc01608b58097324af2c321b00b84c395fbdd58b \
  --commit fcf7665 --out results/experiment_2_4

# 六组并行
bash server/run_e24_formal.sh

# 汇总与图
PYTHONPATH=src venv/bin/python scripts/analyze_experiment_2_4.py
PYTHONPATH=src venv/bin/python scripts/case_analysis_2_4.py
```

| 位置 | 内容 |
|---|---|
| `results/experiment_2_4/G*/seed_*/` | 每个 rollout 的全部日志与 5 个视频 |
| `results/experiment_2_4/summary.csv` | 每组统计 |
| `results/experiment_2_4/video_manifest.csv` | 全部视频索引（组、seed、成功、失败类型、抬升、墙/双指、路径） |
| `results/experiment_2_4/analysis.json` | 分组统计、代表性案例、失败类型 |
| `results/experiment_2_4/case_analysis.json` | 关键案例事件表、首次接触 link、抬升前 EE 运动、Q5 列表 |
| `results/experiment_2_4/figures/` | Figure 1–8 |
| 服务器 | 以上全部 + 约 850 MB 视频（未提交 Git）；180 基础视角 + 60 comparison + 46 events |

## 12. 结论

1. **A→B 的主导失败不是"推不稳"，而是交接后的"抓+搬"**：G0 7/10、G1 9/10 为 `grasp_lift`；双指接触、桌面丢失、无墙接触，EE 目标与实际上升早于物体抬升。
2. **yaw0 是独立的第二失败链**：G2/G4 以撞墙、翻倒、推转为主；退 HOME 只能修复机器人，修不了物体朝向。
3. **同一 B 在 yaw90 独立起点上 9/10 成功**，说明原子推动能力存在；差异来自交接状态下的夹爪-物体相对位姿与腕部构型。
4. **本轮的局部 5 cm 退让无效**（0/10，5/10 完全不接触），但这是固定偏移的失败，不是对所有局部调整的否定。
5. **没有证据支持评测/动作队列错误**：切换、语言、动作维度、判据都按 v2.2 复核；G3 的成功是同一 B 策略。

因此，下一轮应优先解决"交接构型 → 抓取模式"的状态依赖（例如独立的预接触状态族、后继可行性引导），而不是继续扩大随机关节范围；yaw0 则需要单独的接触丰富数据或更稳的推动策略。
