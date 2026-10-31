# 实验 3 受控诊断 v2：启动状态、连续 transition 与冻结 D2 行为

## 0. 结论先行

在 **ep064 f000** 上完成 25 次受控 rollout：

| 条件 | 启动 | v2.2 成功 | wall contact |
|---|---|---:|---:|
| **A snapshot** | B 参考场景 object_B | **5/5** | 0/5 |
| **B snapshot** | A 场景 object_A，qB 诊断放置 | **5/5** | 0/5 |
| **C snapshot** | A 场景，恢复规划终态快照 | **4/5** | 0/5 |
| **B continuous** | A 场景，qB 放置后同一进程启动 D2 | **5/5** | 0/5 |
| **C continuous** | A 场景，真实 q_A→q_B 后**不恢复快照**，同进程启动 D2 | **5/5** | 0/5 |

**最重要的结果：**

- **C continuous 与 B continuous 的 5 个配对 seed 结果完全相同，第一次 action 的七关节 L2 差为 0.0，最终 gap/tilt 也完全相同。** 这说明在这个状态上，真实连续 transition 没有留下可测量的额外残余。
- **B snapshot 与 C snapshot 的实际首推理状态几乎完全一致**（q L2 9.1×10⁻⁷ rad，相对位姿平移 1.9×10⁻⁴ mm，top/wrist MAE=0），第一次 action 七关节 L2 也**精确为 0.0**；但 C snapshot 的 seed4 失败（tilt 19.78°），B snapshot 成功。二值成功对微观数值/动力学差异敏感，**不能归因于 transition history**。
- **A/B 只有 1.864 mm 物体位置差 + top MAE 0.058**；A 5/5、B 5/5，首动作七关节 L2 0.0084–0.0103 rad。没有观察到成功差异，但样本只有 5 个 seed，不能宣布物体位置无影响。
- **v2 的 25 次中 wall_contact 为 0**；24 次 v2.2 成功。v2.2 成功不是墙接触成功。
- 之前实验 3 的 AtoB 3/9 vs Bref 6/9 未复现；旧 runner 没有显式策略种子，A row seed 53050005 vs B/C 64，随机流不同。

---

## 1. 本轮修正的验收问题

| 审查问题 | 修复 |
|---|---|
| `ee_obj_rel_mm` 只复制物体位置差 | 现在计算完整 `T_EE⁻¹ T_object`：相对平移 + 相对四元数旋转；`rel_pos_mm` / `rel_rot_deg` |
| 首动作只取 q1,q4,q7 | 现在读取完整 `first action q1..q7 + grip`，计算七关节 L2 |
| 离线审计 30 步 vs rollout build 2 步 | 统一为 **30-step settle**；直接读取每个 rollout 首推理前保存的状态/图像，不再使用独立离线状态 |
| C 组只是恢复终态快照 | 新增 **C continuous**：A 终态→真实规划→同进程立即启动 D2，不恢复快照 |
| 相同二值结果不代表行为相同 | 报告 gap/tilt/first action/连续结果，不再只看 success |
| 15 次原始 trace 未归档 | `e3c_v2_traces.tar.gz`（25 次轻量 trace，排除视频）+ observation hashes + manifest |

---

## 2. 受控设计

目标 B：**ep064 f000**，在看到新结果前固定。  
统一：D2 checkpoint、B90 指令、相机、max ticks=260、v2.2 判据、显式配对 seed 0–4。

- **A snapshot**：B 参考场景，qB，object_B。
- **B snapshot**：A 场景，qB diagnostic reset，object_A；30-step settle。
- **C snapshot**：A 场景，恢复先前 transition endpoint；30-step settle。
- **B continuous**：A 场景；reset qB；30-step settle；同一进程启动 D2。
- **C continuous**：A 场景；执行 `q_A→q_B` 规划；设置 grip；30-step settle；**同一进程、不恢复快照**启动 D2。

A/B 物体位置差 **1.864 mm**；B/C 同一物体。

---

## 3. 实际首推理状态对齐

完整表：`state_alignment.md`、`data/state_alignment_v2.csv`。

### B/C snapshot

| 变量 | 值 |
|---|---:|
| q L2 | **9.07×10⁻⁷ rad** |
| q max | 5.26×10⁻⁷ rad |
| dq L2 | 6.55×10⁻¹¹ |
| EE position | **1.79×10⁻⁴ mm** |
| EE rotation | 0° |
| gripper | **1.88×10⁻⁶ mm** |
| object position | **3.61×10⁻⁸ mm** |
| **relative translation** | **1.91×10⁻⁴ mm** |
| **relative rotation** | **0°** |
| top RGB MAE | **0.0** |
| wrist RGB MAE | **0.0** |
| contacts | 0 vs 0 |

### B/C continuous

| 变量 | 值 |
|---|---:|
| q L2 | **0.0** |
| EE position | **0.0** |
| relative translation | **0.0** |
| relative rotation | **0.0277°** |
| top/wrist MAE | **0.0** |
| contacts | 0 vs 0 |

### A/B

| 变量 | 值 |
|---|---:|
| q L2 | 5.29×10⁻⁸ rad |
| EE position | 3.33×10⁻⁵ mm |
| object position | **1.864 mm** |
| relative translation | **1.864 mm** |
| relative rotation | **177.46°**（块体 180° 对称；axis 2.54°） |
| top RGB MAE | **0.05798** |
| wrist RGB MAE | 0.0 |

**B/C 满足全部记录阈值；A/B 的完整物理场景不匹配。**  
`q_target` 字段记录的是实际保持 q，不是独立命令目标；真实 D2 动作目标存于 `first_action_v2.csv`。这是本轮未完全覆盖的控制器内部变量。

![Figure 1](figures/fig01_state_alignment.png)

---

## 4. 结果

### 4.1 分组

| 条件 | n | v2.2 | wall | mean gap | mean tilt |
|---|---:|---:|---:|---:|---:|
| A snapshot | 5 | **5** | 0 | 15.53 mm | 1.62° |
| B snapshot | 5 | **5** | 0 | 12.47 mm | 0.019° |
| C snapshot | 5 | **4** | 0 | 13.31 mm | 4.19° |
| B continuous | 5 | **5** | 0 | 16.33 mm | 0.152° |
| C continuous | 5 | **5** | 0 | 16.33 mm | 0.152° |

- **C snapshot 唯一失败：seed4，tilt 19.78°**；B snapshot seed4 成功。二者首推理状态几乎相同、首动作七关节 L2 为 0.0。失败来自**后续动力学/微观数值放大**，不是 transition history。
- **B/C continuous 五次完全一致**：gap/tilt 相同到表内精度，首动作 L2=0。
- **wall_contact 0/25**：24 次 v2.2 成功都不是真实墙接触。12 次 snapshot 成功中 A/B 5+5，C4；连续10；总24。

![Figure 2](figures/fig02_paired_outcomes.png)
![Figure 4](figures/fig04_success_contact.png)

### 4.2 首动作

完整七关节动作 L2（`summaries/first_action_diff_v2.csv`）：

| pair | 范围 |
|---|---:|
| A vs B snapshot | **0.00836–0.01029 rad** |
| B vs C snapshot | **0.0** |
| B vs C continuous | **0.0** |

- B/C 的首动作**完全相同**；这比旧版的 0.0015 rad 差更干净。
- A/B 的首动作差约 0.01 rad；A/B 成功均 5/5，不能识别物体位置效应。

![Figure 3](figures/fig03_first_action_diff.png)

### 4.3 seed4 的 B/C snapshot 差异

| seed | A | B snap | C snap | B cont | C cont |
|---:|---|---|---|---|---|
| 0 | ✅ | ✅ | ✅ | ✅ | ✅ |
| 1 | ✅ | ✅ | ✅ | ✅ | ✅ |
| 2 | ✅ | ✅ | ✅ | ✅ | ✅ |
| 3 | ✅ | ✅ | ✅ | ✅ | ✅ |
| 4 | ✅ | ✅ | ❌ (tilt 19.78°) | ✅ | ✅ |

C snapshot 失败而 B snapshot 成功，虽然：
- q 差 9.1×10⁻⁷ rad；
- 物体差 3.6×10⁻⁸ mm；
- 图像 MAE 0；
- 首动作 L2 0。

这本身是一个重要结果：**二值成功不能作为“启动状态等价”的敏感检验**；后续动力学可放大极小差异。连续 B/C 没有这种差异，支持“不是 transition 残余”，但 seed4 的放大机制仍未完全识别。

---

## 5. 对旧结果的修正

旧实验 3：
- AtoB 3/9，Bref 6/9；
- 无显式 torch seed；
- A row seed 53050005，B/C 64；
- B Reference 是重建场景。

受控 v2：
- A 5/5，B 5/5，C snapshot4/5；
- B/C continuous 5/5 且完全一致。

**旧 3/9 vs6/9 不能作为 transition 因果证据。** 新结果也不支持“transition 导致额外失败”。

---

## 6. 八个问题

**1. 规划能否可靠到达 B90 记录构型？**  
Pilot 3/3，q L2 ≤8×10⁻⁶ rad，EE ≤0.0037 mm；连续 C 也能在真实执行后启动 D2。只有 3 个 B 目标、1 个唯一 A 终态。

**2. 匹配 q 后哪些变量仍未匹配？**  
- B/C：没有记录到的差异；相对位姿 1.9×10⁻⁴ mm / 0°，图像 0。
- A/B：物体 1.864 mm、相对平移 1.864 mm、完整相对旋转 177.46°（axis 2.54°）、top 0.058。
- 控制器独立命令目标未完全保存；仅动作 target 保存。

**3. B/C 是否具备可比较完整启动状态？**  
**是（记录变量与阈值内）**，且首动作 L2=0。  
连续 B/C 也完全一致；仍有未记录的 controller 内部状态，但当前没有证据显示其影响。

**4. 物体场景改变是否伴随 D2 首段动作变化？**  
A/B 首动作七关节 L2 **0.0084–0.0103 rad**；A/B 均 5/5。  
**有小的首动作差，但未观察到成功差异**；n=5，不能识别物体位置的独立效应。

**5. 是否观察到 transition 相关残余？**  
**连续 C vs B：没有**（5/5 相同，首动作 0，gap/tilt 相同）。  
Snapshot C 的 seed4 失败与状态/首动作无关，属于数值/动力学放大；不能归因于 transition。

**6. B90 参考起点本身多稳定？**  
B snapshot 5/5，A snapshot5/5；B/C continuous 5/5。  
但 C snapshot seed4 失败、历史 Bref 6/9，说明二值成功对 seed/微观状态敏感；5 seeds 不足以给稳定边界。

**7. 哪些差异可能解释首次接触错误？**  
- seed4 C snapshot 失败：tilt19.78°，首次接触仍 robot_link10，不是墙；启动状态与 B 几乎相同，首动作相同，故不是首次接触因状态差异直接改变。
- A/B 物体差1.864mm/top0.058是候选，但无成功差异。
- 当前不能识别 seed4 的动力学放大机制。

**8. 下一步最值得单独控制什么？**  
**先控制数值/动力学重复性和物体场景**：
- 同状态重复运行、固定 solver/seed，量化 seed4 类放大；
- **object 完全相同的 A/B**（显式物体重定位作为单独变量）；
- 原始训练 RGB vs 重建 RGB；
- 记录真实控制器目标（不仅 action）和内部队列；
- 增加配对 seed/独立 B 目标。

---

## 7. Observed / Supported / Not Identifiable / Weakened

### Observed

- B/C snapshot 首推理 q 9.1×10⁻⁷ rad，相对平移1.9×10⁻⁴ mm，图像0，首动作七关节L2=0。
- B/C continuous 5/5 完全相同；首动作0；gap/tilt相同。
- C snapshot seed4 失败（tilt19.78°），B snapshot seed4 成功。
- A/B 物体1.864mm，top0.058；首动作0.0084–0.0103 rad；均5/5。
- 25 次中 wall_contact=0，v2.2 成功24/25。

### Supported

- 在连续 transition 下，B/C 没有可测量的额外状态或行为差异。
- 旧 3/9 vs6/9 至少部分来自未配对随机性；受控后不复现。
- 二值成功对微观差异敏感；不能把相同 success 当作行为相同。
- v2.2 success 与 wall contact 必须分开。

### Not Identifiable

- C snapshot seed4 失败的具体动力学放大机制（状态和首动作相同）。
- A/B 的1.864mm物体差与top0.058的独立成功效应（n=5）。
- 控制器内部目标/队列对长期行为的贡献。

### Contradicted / Weakened

- “AtoB 3/9 vsBref6/9 证明 transition 有害”——受控未复现。
- “q 匹配足以恢复 B90”——B/C 5/5 支持该状态，但 seed4 snapshot 显示二值结果仍受微观差异影响；不能外推。
- “v2.2 成功=墙接触”——0/25 wall contact。

---

## 8. 完整 A→B 视频

`videos/AtoB_full.mp4`（1280×520，20 fps）：

- 第一段：A90 终态 → 规划过渡（44 帧展示）；
- 第二段：**C continuous**（真实 transition 后立即启动 D2，seed0）100 个 policy tick；
- 左侧 side，右侧 top；包含接近、接触、推动。

![Video frames](figures/fig05_ato_b_video_frames.png)

---

## 9. 限制

- 1 个 B 目标 ×5 配对 seed；不是统计泛化。
- 1 个唯一 A 终态。
- B reference 是重建场景，不是原始训练快照。
- **控制器独立命令目标仍未完整保存**；`q_target` 是实际保持 q；D2 action target 已保存。
- 连续 B/C 只在 ep064 f000 验证；不能外推到其他 B 起点。
- 规划为碰撞采样直线；执行位置命令，不是速度/加速度连续时间控制器。
- wall_contact 基于仿真接触日志；未做力/穿透审计。
- C snapshot seed4 的微观放大机制未识别。
