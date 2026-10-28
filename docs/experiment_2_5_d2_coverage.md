# 实验 2.5 补充诊断：C3 启动状态与 D2 训练状态的最近邻覆盖（状态对齐修正版）

## 摘要与直接回答

上一版把 **C3 第一次 B 动作的夹爪目标（0.01656 m）** 与 **D2 ep66 f39 的观测夹爪开度（0.014 m）** 比较，得到 2.56 mm。这是字段混用。C3 第一次 B 推理前的真实观测夹爪是 **0.040 m**；它与 ep66 f39 的观测差是 **26.0 mm**，与 D2 B 训练起点（0.035 m）差 **5.0 mm**。修正后：

- **Q1：是否混用了？** 是。旧报告的 2.56 mm 是 **action target vs observation** 的差；真实观测差是 26.0 mm（对 ep66 f39）。
- **Q2：C3 与 ep66 f39 的真实夹爪观测差？** **26.0 mm**（0.040 vs 0.014 m）。
- **Q3：加入真实夹爪后，还有完整观测近邻吗？**
  - 训练起点：ep74，q 0.9136 rad、夹爪差 5.0 mm，但 EE 差 152.4 mm、相对几何差 149.5 mm → **不是完整观测近邻**。
  - 训练中间帧：ep66 f39，q 0.1710 rad、EE 差 7.7 mm、相对几何差约 1.9 mm，但夹爪差 **26.0 mm** → **关节/相对几何近邻，不是完整观测近邻**。
  - 夹爪 ≤5.1 mm 的中间帧：**0 个**；≤10 mm 的中间帧只有每条示范第 1 帧（夹爪 0.032 m），q 仍 0.914 rad。
  - **没有任何一帧同时满足 q<0.5 rad、夹爪差≤10 mm，也没有 q<0.5 rad、相对几何<50 mm、夹爪<10 mm 的完整近邻。**
- **Q4：专家与 C3 VLA 的方向差异还成立吗？** 作为描述性对照仍成立：ep66 f39 之后专家 EE z 从 0.070 降到 0.018 m，C3 VLA 前 2–6 tick 升到 0.092 m。但两者初始夹爪观测差 26 mm，不能再说“从几乎完全相同的物理状态出发”；该对照的因果强度必须降低，只能作为“相似关节/末端几何、不同夹爪观测下的动作差异”。
- **Q5：最准确的覆盖描述？**
  - Joint-State Coverage：**有**（ep66 f39，0.171 rad；ep22 f39，0.118 rad 但 yaw0）。
  - Relative Geometry Coverage：**近似有**（ep66 f39 约 1.9 mm，但用起点物体位置重建，不是逐帧真值）。
  - Gripper-State Coverage：**没有**（ep66 f39 差 26 mm；B 起点差 5 mm，但关节/EE 远）。
  - Full Observation Coverage：**没有**。
  - Policy Launch Capability：**未证明**；C3 VLA 从该状态启动的运动方向与专家不同，但初始夹爪观测不同，不能单独归因。

---

## 1. State Alignment Audit（字段、时间点、单位、语义）

C3 的控制顺序（`scripts/run_experiment_2_5.py::Episode25.policy_tick`）：

```python
frames = self.capture()                  # 渲染观测
state  = concat(q, finger_widths)        # 本体观测（9 维）
action = self.runner.act(frames, state, task)  # 模型推理
command(action[:7]); set_gripper(action[7])    # 发送目标
for 12 physics steps: step()
```

因此：

| 字段 | 来源 | 时间点 | 语义 | 单位 |
|---|---|---|---|---|
| `q1..q7` | `policy_trace.state[0:7]` | **B 第一次推理前** | 实际关节位置 | rad |
| `finger1/2` | `policy_trace.state[7:9]` | 同上 | 两根手指实际位置 | m |
| `action[7]` | `policy_trace.action[7]` | 同上，**推理输出** | 夹爪目标（绝对位置） | m |
| `executed_gripper_target` | `policy_trace` | 同上，实际发送 | 控制器目标 | m |
| `grip_actual` | `physics_trace.csv` 同一 `physics_step` | 动作执行前 | 手指实际位置 | m |
| D2 `observation.state[7:9]` | parquet | 帧 t，**观测** | 两根手指实际位置 | m |
| D2 `action[7]` | parquet | 帧 t，**动作目标** | 夹爪目标 | m |

关键值：

| 项 | 值 |
|---|---:|
| C3 B 第一次观测 `finger1/finger2` | **0.040 / 0.040 m** |
| C3 B 第一次动作 `action[7]` | **0.01656 m** |
| C3 同一 `physics_step` 的 `grip_actual` | **0.040 m** |
| D2 B 训练起点观测（120 条） | **0.035 m**（每指都是 0.035） |
| D2 B 训练起点第一个动作目标 | **0.014 m** |
| D2 ep66 f39 观测 | **0.014 m** |
| D2 ep66 f39 动作目标 | **0.014 m** |

**时间对齐结论**：`policy_trace` 的 state 是推理前观测，action 是推理后目标；D2 的 observation 与 action 同在帧 t，但语义不同。旧报告把 C3 的 action 与 D2 的 observation 比较，属于字段混用，不是时间偏移。

## 2. 修正前后对照

| 项 | 旧报告 | 修正后 |
|---|---:|---:|
| C3 夹爪值 | 0.01656 m（动作目标） | **0.040 m（观测）** |
| 对比 D2 ep66 f39 | 0.014 m（观测） | 0.014 m（观测） |
| 夹爪差 | 2.56 mm | **26.0 mm** |
| 对比 D2 B 起点 | 未比较 | 0.035 m，差 **5.0 mm** |
| ep66 f39 是否“完整状态近邻” | 是 | **否**（仅关节/相对几何近） |
| 0.171 rad 关节近邻 | 成立 | **仍成立，未改变** |
| 1.87 mm 相对几何 | 成立 | **仍成立，但用起点物体位置重建，非逐帧真值** |

## 3. 重新计算：起点与中间帧

数据：`data/_shards_e23/B_D2`，120 episodes / 14,653 frames；与 `D2_episodes.jsonl` 的 120 条成功起点最大对齐残差 1.6e-7 rad。

### 3.1 Initial-State NN（真实观测）

| C3 seed | q 最小 | episode | start type | 夹爪观测差 | EE 差 | 相对几何差 | yaw 差 | 墙间隙差 |
|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 40000000 | 0.913643 | 74 | diverse | 5.0 mm | 152.4 mm | 149.5 mm | 2.19° | 3.35 mm |
| 40000001 | 0.912133 | 74 | diverse | 5.0 mm | 152.4 mm | 149.5 mm | 2.19° | 3.35 mm |
| 40000002 | 0.907898 | 74 | diverse | 5.0 mm | 152.4 mm | 149.5 mm | 2.19° | 3.35 mm |

**起点结论**：夹爪观测接近（5 mm），但关节构型、EE 位置、相对物体几何都远；**不是完整观测近邻**。

### 3.2 Trajectory/Interior NN（真实观测）

| C3 seed | interior 最小 | episode/frame | yaw 几何 | q | 夹爪观测 | 夹爪差 |
|---:|---:|---|---:|---:|---:|---:|
| 40000000 | **0.118369** | ep22 f39 | yaw0（不匹配） | 0.118 | 0.014 | 26 mm |
| 40000000 | **0.171040** | **ep66 f39** | **yaw90（匹配）** | 0.171 | 0.014 | **26 mm** |
| 40000001 | 0.119723 / 0.172844 | ep22 f39 / ep66 f39 | yaw0 / yaw90 | — | 0.014 | 26 mm |
| 40000002 | 0.116190 / 0.171860 | ep22 f39 / ep66 f39 | yaw0 / yaw90 | — | 0.014 | 26 mm |

ep66 f39 物理对比（seed 40000000）：

- q 0.1710 rad；
- EE 世界位置差 **7.70 mm**；
- EE–物体相对位置差 **1.87 mm**；
- 物体 yaw 差 **2.19°**；
- 墙间隙差 **1.73 mm**；
- **夹爪观测差 26.0 mm（0.040 vs 0.014）**。

### 3.3 夹爪受限搜索

| 容差 | 子集 | 帧数 | 不同 episode | 最近 q | 最近帧 |
|---:|---|---:|---:|---:|---|
| 2 mm | 全部 / interior / yaw90 | 0 | 0 | — | — |
| 5.1 mm | 起点 | 120 | 120 | 0.9136 | ep74 f0 |
| 5.1 mm | interior | **0** | 0 | — | — |
| 10 mm | interior | 120 | 120 | 0.9136 | ep74 f1（夹爪 0.032，差 8 mm） |
| 10 mm | interior yaw90 | 30 | 30 | 0.9136 | ep74 f1 |

**没有中间帧的夹爪观测接近 C3 的 0.040 m**：B 示范从第一步就闭到 0.014 m，唯一接近的是起点的 0.035 m（差 5 mm），而起点关节/EE 远。

## 4. 专家 vs C3 VLA（降低强度的描述性对照）

ep66 f39 之后，专家（20 Hz）：

```text
t=0  EE z 0.0700  obs grip 0.014  action grip 0.014
t=2  EE z 0.0486  obs grip 0.014  action grip 0.014
t=4  EE z 0.0258  obs grip 0.014  action grip 0.014
t=6  EE z 0.0182  obs grip 0.014  action grip 0.014
```

C3 VLA 从 B 第一次 tick 起（tick 0 为执行动作前）：

```text
t=0  EE z 0.0696  obs grip 0.040  action grip 0.01656
t=2  EE z 0.0917  obs grip 0.040  action grip 0.0174
t=4  EE z 0.0820  obs grip 0.040  action grip 0.0166
t=6  EE z 0.0757  obs grip 0.040  action grip 0.0167
```

- 专家：下降 5.2 cm，建立推动接触；
- C3 VLA：前 6 tick 先上升 2.2 cm，随后才下降；
- 动作目标：C3 0.01656 vs 专家 0.014（差 2.56 mm），**动作层面接近**；
- **观测**：C3 0.040 vs 专家 0.014（差 26 mm），**观测层面不接近**。

因此运动方向差异仍然存在，但不能再说“从几乎相同的完整状态出发”；它只能作为“相似关节/末端几何、不同夹爪观测”的描述性证据。初始本体观测不同，可能影响策略；不能把方向差异单独归因于模型忽略视觉或阶段。

## 5. 实际模型输入

| 输入 | 是否可用 | 说明 |
|---|---|---|
| D2 ep66 f39 top/wrist 256×256 | **有** | `matched_frames/d2_ep66_{top,wrist}_f39.png`，原始训练视频抽帧 |
| C3 B 第一次推理的 256×256 top/wrist | **没有** | 测试时 `rig256.capture()` 只传给策略，未落盘；现有 `top.mp4` / `contact.mp4` 是 640×480 诊断渲染 |
| C3 top/contact 640×480 | 有 | 同一几何，但不是逐像素 VLA 输入 |
| 语言指令 | 有 | 两者都是 `push the orange block along the positive X direction to the wall` |

- 训练 wrist f39：夹爪闭合（0.014 m），手指间距窄；
- C3 contact f153：夹爪张开（0.040 m），手指间距明显更宽（图可见）。
- 图像看起来相似不代表网络特征相似；这里只核对输入，不解释 attention。

## 6. 五问结论（分类）

| 覆盖类型 | 结论 | 证据 |
|---|---|---|
| **Joint-State Coverage** | **有** | ep66 f39 0.171 rad（yaw90）；ep22 f39 0.118 rad（yaw0） |
| **Relative Geometry Coverage** | **近似有** | ep66 f39 相对位置差 1.87 mm（用起点物体位置重建，非逐帧真值） |
| **Gripper-State Coverage** | **没有** | ep66 f39 差 26 mm；起点差 5 mm 但关节/EE 远；interior ≤5.1 mm 为 0 帧 |
| **Full Observation Coverage** | **没有** | 关节/相对几何近的帧夹爪差 26 mm；夹爪近的起点关节差 0.914 rad、EE 152 mm |
| **Policy Launch Capability** | **未证明** | C3 VLA 0/3；从 ep66 f39 类似状态启动时方向与专家不同，但初始夹爪观测不同，不能单独归因 |

**最准确描述**：

> C3 的启动状态在**关节空间和近似相对几何**上被 D2 的 **单条 yaw90 示范 ep66 f39** 覆盖；但**夹爪观测不覆盖**（26 mm），因此**完整观测不覆盖**。冻结 VLA 从该状态启动时，动作目标虽然接近专家（2.56 mm），但垂直运动方向与专家相反；由于初始夹爪观测不同，这一方向差异只能作为描述性证据，不能证明模型在“完全相同的观测”下失败。

## 7. 交付与复现

- 报告：本文件
- 数据/CSV：`results/experiment_2_5/d2_c3_coverage/`
  - `c3_query_states.csv`
  - `initial_nn.csv`（含 `grip_obs`、`grip_diff_mm`）
  - `interior_nn.csv`、`interior_nn_yaw90.csv`（含夹爪字段）
  - `gripper_constrained_nn.csv`（2 / 5.1 / 10 mm）
  - `all_frame_nn.csv`、`episode_deduplicated_nn.csv`
  - `nearest_trajectory_comparison.csv`、`..._yaw90.csv`
  - `analysis_summary.json`（含 `state_alignment_audit`）
- 图：`figures/fig1..fig7`；`fig7_state_alignment.png` 为观测/动作夹爪字段对照。
- 图像：`matched_frames/`
- 脚本：`scripts/analyze_d2_c3_full.py`（已加入 `grip_obs`、`grip_diff_mm`、夹爪受限搜索、state audit）
- 原始训练数据：`data/_shards_e23/B_D2`（120 eps / 14,653 frames）

## 8. 仍无法核实

- D2 parquet 没有逐帧物体位姿和接触标签；ep66 f39 的“相对几何 1.87 mm”用起点物体位置近似，不是逐帧测量真值。
- C3 的 256×256 VLA 输入没有落盘；只能看到 640×480 诊断图，不能做像素级输入对比。
- 只有 3 个 C3 seed；task-compatible 训练轨迹只有 ep66 一条。
- 没有视觉嵌入/attention 分析。
