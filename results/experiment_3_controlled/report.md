# 实验 3 受控诊断 v3：目标到达验证后的 B/C 连续 transition

## 0. 结论先行

v2 的 continuous 组虽然 B/C 完全相同，但**没有到达 qB**：启动 q 与 qB 差约 0.671 rad，EE 差约 207 mm，q6 速度约 −17.64 rad/s。因此 v2 continuous **不能作为 transition 证据**。

v3 修正为**目标到达/稳定验证**：

- 在规划执行、夹爪调整、稳定等待、首次 D2 推理前，分别记录 q、dq、EE、夹爪、物体、接触；
- 使用**收敛条件**而不是固定 30 步：q L2≤0.01 rad、q max≤0.01、dq L2≤0.2、物体速度≤0.01 m/s、夹爪误差≤2 mm，连续保持 120 步（0.5 s）；
- 最大等待 1200 步；未收敛则标记 transition failure，不启动 D2；
- 保存真实 q target、dq、首推理图像和**未四舍五入的 raw action**。

v3 结果：

| 条件 | n | v2.2 成功 | wall contact | 目标 q 误差 | dq |
|---|---:|---:|---:|---:|---:|
| A snapshot | 5 | **5/5** | 0 | — | — |
| B snapshot | 5 | **5/5** | 0 | — | — |
| C snapshot | 5 | **4/5** | 0 | — | — |
| **B continuous v3** | 5 | **4/5** | 0 | **5.2×10⁻⁶ rad** | **5.7×10⁻¹³** |
| **C continuous v3** | 5 | **4/5** | 0 | **5.2×10⁻⁶ rad** | **5.7×10⁻¹³** |

- **B/C continuous 都真正到达 qB**：201 步收敛（0.8375 s），q 误差 5.2×10⁻⁶ rad，dq 5.7×10⁻¹³；实际 q/EE/物体/图像一致。
- B/C continuous 的 5 个配对 seed **结果与行为完全相同**：4/5，失败均为 seed4 tilt 19.26°；首动作七关节 L2=0；gap/tilt 相同。
- **没有观察到连续 transition 的额外残余影响。**
- **25 次中 wall_contact=0**；22 次 v2.2 成功，3 次失败（C snapshot seed4、B/C continuous seed4）。v2.2 成功不是墙接触。

---

## 1. v2 的问题与 v3 修复

| 问题 | v2 | v3 |
|---|---|---|
| continuous 目标到达 | 只 `sim.command(qb)` + 30 步（0.125 s）；实际 q 与 qB 差 **0.671 rad**，EE 差 **207 mm**，q6 dq **−17.64 rad/s** | 收敛循环，最大 1200 步；201 步达到 q 误差 **5.2×10⁻⁶ rad**，dq≈0 |
| 目标验证 | 无 | `transition_convergence.csv` + `converged` 标志 |
| 未收敛处理 | 仍启动 D2 | **不启动 D2**，标记 transition failure |
| `q_target` | 写入当前实际 q，不是命令目标 | 写入真实 qB |
| 首动作精度 | `policy_trace` 5 位小数 | **raw_actions.npz 全精度** |
| C 连续执行 | 5/5 但未到目标，无效 | 4/5，目标到达有效 |
| 归档 | v2 trace | v3 trace `e3c_v3_traces.tar.gz` + hashes |

收敛定义：

- q L2 ≤0.01 rad
- q max ≤0.01 rad
- dq L2 ≤0.2
- object speed ≤0.01 m/s
- grip error ≤2 mm
- 连续满足 120 步（0.5 s）
- 最大 1200 步（5 s）

---

## 2. v3 continuous 目标到达

`data/b90_rollout_results_v3.csv`：

| group | seed | converge | steps | q_err(qB) | dq_norm | success | gap | tilt |
|---|---:|---|---:|---:|---:|---|---:|---:|
| B | 0 | ✅ | 201 | 5.2e-6 | 5.7e-13 | ✅ | 16.83 | 0.005 |
| B | 1 | ✅ | 201 | 5.2e-6 | 5.7e-13 | ✅ | 17.20 | 1.035 |
| B | 2 | ✅ | 201 | 5.2e-6 | 5.7e-13 | ✅ | 18.35 | 0.014 |
| B | 3 | ✅ | 201 | 5.2e-6 | 5.7e-13 | ✅ | 13.39 | 0.019 |
| B | 4 | ✅ | 201 | 5.2e-6 | 5.7e-13 | ❌ | 11.37 | 19.26 |
| C | 0–4 | ✅ | 201 | 5.2e-6 | 5.7e-13 | 4/5 | 同 B | 同 B |

- B/C 的 q、dq、q_target、grip、物体、图像、首动作均一致；`state_alignment_v3.csv` 中 q/EE/relative pose/action7 全部为 **0**，top hash 相同。
- B/C seed4 同时失败，说明不是 C 额外引入；是该状态/seed 下策略动力学本身失败。
- **C 到达 qB 后 B/C 完全相同**，这是本轮真正的“连续 transition”证据。

![Figure 1](figures/fig01_state_alignment.png)
![Figure 4](figures/fig04_success_contact.png)

---

## 3. B/C 启动状态

`state_alignment_v3.csv`（连续）：

| seed | q L2 | EE mm | relative pos mm | relative rot deg | grip mm | top MAE | action7 L2 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0–4 | **0.0** | **0.0** | **0.0** | **0.0** | **0.0** | **0.0** | **0.0** |

- top hash 相同，wrist 相同。
- 目标 q、实际 q、q_target 均为 qB（差异 5.2e-6），dq≈0。
- B/C continuous 是当前最干净的状态匹配对照。

A/B snapshot 仍是场景改变对照：物体位置差 **1.864 mm**，top MAE **0.058**；首动作七关节 L2 **0.0084–0.0103 rad**；A/B 都 5/5。

---

## 4. Snapshot 的 B/C 差异

Snapshot 组：
- B 5/5，C 4/5；
- C seed4 tilt **19.78°**；
- B/C 首推理 q 差 **9.1×10⁻⁷ rad**，相对位姿 1.9×10⁻⁴ mm，图像0，首动作 L2=0。

C seed4 失败与 B/C 首动作相同并存，说明二值成功对后续动力学/微观数值高度敏感。  
**不能把这个 seed4 差异归因于 transition history**；snapshot C 也经过了快照恢复，而不是连续 transition。v3 continuous seed4 是 B/C 同时失败，进一步支持“不是 C 特有”。

---

## 5. 旧结果与 v2 的处理

- 旧实验 3 AtoB 3/9 vs Bref 6/9：未配对策略种子（A row seed 53050005 vs B/C 64），**不用于因果结论**。
- v2 continuous 5/5：目标 q 误差 **0.671 rad**、EE **207 mm**，**无效**；已被 v3 取代。
- v2 snapshot A/B/C 5/5/5/4、状态对齐仍然有效；v3 保留 snapshot 作为对照。
- v2 continuous 的 5/5 不再作为结论；v3 的 4/5/4/5 才是目标到达后的结果。

---

## 6. 八个问题

**1. 规划能否可靠到达 B90 构型？**  
Snapshot 路径可执行；v3 continuous 通过收敛验证：201 步、q 误差 5.2e-6 rad、dq≈0。不是固定 30 步的假到达。

**2. 匹配 q 后哪些变量仍未匹配？**  
- B/C continuous：**无记录差异**；q/EE/relative pose/action7 全0，图像 hash 相同。
- A/B：物体 1.864 mm、相对位姿 1.864 mm、top 0.058。
- 控制器独立命令目标现已保存为 qB；内部控制器状态仍未完全审计。

**3. B/C 是否具备可比较完整启动状态？**  
**是**：v3 continuous 满足收敛和所有记录阈值；B/C 首动作全精度 L2=0。

**4. 物体场景改变是否伴随首段动作变化？**  
A/B 七关节 L2 **0.0084–0.0103 rad**；A/B 均5/5。  
有小的首动作差，但没有成功差异；n=5，不能识别物体位置独立效应。

**5. 是否观察到 transition 残余？**  
**没有**：B/C continuous 目标一致、首动作一致、5 seed 行为一致；seed4 同时失败。  
Snapshot C seed4 差异不能用 transition 解释，因为 C snapshot 是快照恢复组。

**6. B90 参考起点稳定吗？**  
A/B snapshot 5/5；B/C continuous 4/5（seed4 同失败）；C snapshot4/5。  
二值成功对 seed 敏感；5 seeds 不足以给稳定边界。

**7. 首次接触错误可能来自什么？**  
- v3 所有运行 wall_contact=0；v2.2 成功不要求墙接触。
- seed4 B/C 均失败，tilt19.26°；不是 C 特有。
- 首次接触日志仍为 `robot_link10`（夹爪/手指），不是墙。
- 具体动力学放大机制仍未识别。

**8. 下一步最值得控制什么？**  
**物体场景与原始训练图像**，以及**数值/动力学重复性**：
- object 完全相同的 A/B（显式物体重定位作为单独变量）；
- 原始训练 RGB vs 重建 RGB；
- 增加独立 B 目标与配对 seed；
- 记录控制器内部 target/queue（本轮已有 q_target=qB，但内部状态仍有限）。

---

## 7. Observed / Supported / Not Identifiable / Weakened

### Observed

- v3 B/C continuous 201 步收敛，q 误差5.2e-6 rad，dq5.7e-13；B/C q/EE/relative pose/action7=0，图像 hash相同。
- B/C continuous 5 seeds 均4/5，失败均 seed4 tilt19.26°，gap/tilt相同。
- Snapshot A5/5、B5/5、C4/5；C snapshot seed4 tilt19.78°。
- 25 次 wall_contact=0；22 次 v2.2 成功。
- A/B 物体1.864mm，top0.058，首动作0.0084–0.0103 rad；A/B均5/5。

### Supported

- 连续 transition 到达 qB 后，B/C 没有可测量的额外状态或行为差异。
- v2 continuous 的 5/5 是假到达；v3 修正后是 4/5，且 B/C 同失败。
- 旧 3/9 vs6/9 至少部分来自未配对种子。
- 二值成功不能替代行为/轨迹比较；v2.2成功≠墙接触。

### Not Identifiable

- seed4 在 B/C 中同时失败的动力学放大机制。
- A/B 的1.864mm物体差与top0.058的独立效应（n=5）。
- 控制器内部状态对长期行为的贡献。

### Contradicted / Weakened

- “v2 continuous 5/5 证明 transition 无影响”——目标未到达，**被推翻**。
- “AtoB 3/9 vsBref6/9 证明 transition 有害”——未配对，**被削弱**。
- “q 匹配足以恢复 B90”——B/C 4/5 且 seed4 失败，说明还有策略/动力学因素。
- “v2.2 成功=墙接触”——0/25 wall contact，**被削弱**。

---

## 8. 完整 A→B 视频

`videos/AtoB_full.mp4`：

- A90 终态 → 真实规划过渡（44 帧展示）；
- 随后 **v3 continuous C seed0**，目标收敛到 qB 后启动 D2（84 policy ticks）；
- 左 side，右 top；包含接近/接触/推动。

![Video frames](figures/fig05_ato_b_video_frames.png)

---

## 9. 限制

- 1 个 B 目标（ep064 f000）×5 seeds；不是统计泛化。
- 1 个唯一 A 终态。
- B Reference 是重建场景，不是原始训练快照。
- 收敛阈值是工程阈值；v3 只在 201 步达到，尚未测试其他 B 起点。
- 控制器内部 target/queue 仍未完整审计；本轮已保存 q_target=qB 与 raw action。
- C snapshot seed4 与 continuous seed4 的关系仍未完全识别：两者都失败，但 snapshot C 失败而 B 成功；连续 B/C 同失败。
- 规划为碰撞采样直线，不是连续时间动态可行轨迹。
- wall_contact 基于仿真接触日志；未做力/穿透审计。
