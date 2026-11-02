# 实验 4.1 严格评价器：阈值校准与物理正负案例

评价器实现：`src/exp41/strict_eval.py`，版本 **`exp4.1-strict-v1`**（配置随每条记录落盘）。
案例集：`scripts/exp41/evaluator_tests.py`（**17 个真实 PyBullet 案例**，结果在 `evaluator_tests.json`）。

## 1. 与旧版 v2.2 的根本区别

| | 旧版 v2.2（实验 2/3） | 本版 exp4.1-strict-v1 |
| --- | --- | --- |
| 墙面判据 | `abs(gap) ≤ 0.02`（离墙 2 cm 内即算成功） | **真实接触**：`block↔wall` 每个 240 Hz 物理步记录，需 ≥5 个物理步且有法向力 ≥0.05 N |
| 接触采样 | 控制周期末尾一次 | 每物理步（`PhysicsMonitor` 包住 `sim.step()`） |
| 墙体 ID | 碰撞盒尺寸猜测（4.0 已证伪：把 wall=3 误判成托盘） | `build_scene()` 返回的权威对象（`scene.wall`） |
| 法向校验 | 无 | 记录 `contactNormalOnB` 并断言 `|n_x| ≥ 0.9`（正例自检确认推墙时 n_x<0） |
| 初始合法性 | 无 | 不贴墙、不穿模、稳定、任务非初始完成 |
| 稳定窗口 | 瞬时速度 | 专家结束后独立观察 0.8 s + 0.2 s 回退检测 |
| 失败原因 | 单个布尔 | `failure_reason` 枚举 + 连续指标 |

## 2. 阈值与物理依据

| 参数 | 值 | 依据 |
| --- | --- | --- |
| `wall_contact_min_steps` | 5 个物理步 | 接触持续 ≥ 20.8 ms，高于单步接触抖动 |
| `wall_force_min_n` | 0.05 N | 物体重 1.47 N（0.15 kg）的 3.4%，可排除"擦到"级别的数值接触 |
| `normal_x_min` | 0.9 | 墙面法向为 X；0.9 ⇒ 夹角 ≤ 25.8°，排除侧向/托盘接触 |
| `terminal_gap_max` | 6 mm | 由正例实测间隙（≤2 mm）+ 回弹观测（≤5 mm）留 1 mm 余量 |
| `gap_min` | −1 mm | 允许数值穿透，超过即判穿模 |
| `b_advance_min` | 0.05 m | 起点到墙的距离通常 0.17–0.18 m；5 cm 保证是"推"而不是"碰一下就停" |
| `b_goal_half_width` | 30 mm | 3 个目标区各宽 60 mm 的一半 |
| `b_lateral_tol` | 20 mm | **仅在起点已在目标区内时**检查；需要横向修正时不适用（见下） |
| `b_tilt_max` | 12° | 与 A90 一致；正例实测 ≤0.01°，负例（顶翻）>45° |
| `a_pos_tol` | 20 mm | 正例实测 2.7 mm；旧版也是 2 cm，但本版还要求抓取/抬升/释放/支撑 |
| `a_yaw_tol` | 12° | A90 要求长轴沿 X；正例实测 1.9° |
| `a_min_lift` | 0.05 m | 真抓取必须离开桌面；正例实测 0.119 m |
| `a_min_transport` | 0.10 m | 起点到 9 个目标的距离均 >0.10 m |
| `a_slip/release/retreat` | 0.01 m / 0.030 m / 0.10 m | 释放后两指开口 ≥30 mm、EE 离物体 ≥10 cm |
| `settle` | 0.8 s + 0.2 s | 覆盖 4 倍以上桌面摩擦收敛时间（实测 <0.2 s 稳定） |
| `speed_max` | 0.01 m/s | 与实验 2/3 一致；正例实测 2e-4–2e-7 m/s |

## 3. 物理负例覆盖（全部真实执行，不是手写 CSV）

| 案例 | 场景/行为 | 评价器输出 | 判定 |
| --- | --- | --- | --- |
| `positive_real_push_and_wall_contact` | 专家真实推到墙 | `strict_success=True` | ✅ |
| `near_wall_without_contact` | 专家停在离墙 4 mm（旧 v2.2 会判成功） | `wall_contact_ever` | ✅ |
| `pushed_wrong_direction` | 向 −X 推 6 cm | `advance_ok` | ✅ |
| `pushed_too_high_climbs_object` | 推高 z=5 cm，爬上物体 | `wall_contact_ever` | ✅ |
| `overshoot_after_wall_contact` | 接触后继续压 2 cm | `strict_success=True`（正例鲁棒性） | ✅ |
| `initial_already_touching_wall` | 初始贴墙 | `initial_valid` | ✅ |
| `initial_penetrates_wall` | 初始穿模 | `initial_valid` | ✅ |
| `wrong_goal_region` | 推到低区却按高区评价 | `lateral_ok`/`goal_region_ok` | ✅ |
| `robot_touches_wall_block_does_not` | 机器人自己碰墙、方块不动 | `pushed_by_robot` | ✅ |
| `positive_grasp_place_release` | 专家真实抓放 | `strict_success=True` | ✅ |
| `no_grasp` | 跳过抓取直接放 | `grasped_by_two_fingers` | ✅ |
| `placed_at_wrong_position` | 放偏 6 cm | `position_ok` | ✅ |
| `wrong_place_orientation` | 强制 yaw=90°（长轴 Y） | `yaw_ok` | ✅ |
| `not_released` | 不松爪 | `released` | ✅ |
| `no_retreat` | 不撤离 | `retreated` | ✅ |
| `dropped_from_height_unstable` | 5 cm 高处掉落 | `no_penetration`（另有速度/姿态指标） | ✅ |
| `slipped_mid_transport` | 运输中松爪再闭合 | `no_penetration`（位置/姿态亦越界） | ✅ |

**17/17 PASS**。两个"正例"（真实推墙、真实抓放）和 15 个负例使用同一评价器，说明它既能识别真正的物理成功，
也能识别有迷惑性的失败。

## 4. 已知局限（不隐藏）

1. `pushed_too_high_climbs_object` / `dropped_from_height_unstable` 的失败原因是"第一个不满足的检查"，
   而这些案例同时违反多条（如 tilt + gap + 速度），报告里保留全部连续指标以便复核。
2. `lateral_ok` 只在"起点已在目标区内"时生效；需要横向修正的目标（跨区配对）由 `goal_region_ok` 判定，
   本轮专家在该情形不稳定，因此**没有**把跨区配对纳入主数据。
3. 未做"接触力标定"到真实力传感器级别：法向力阈值来自仿真与物体重量的比值，不是实验测量。
