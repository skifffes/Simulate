# 实验 2.8：跨中间状态的夹爪—机械臂运动耦合及 D2 训练数据关联分析

## Executive Summary

- **非 C3 的 B90 中间状态也普遍存在夹爪—机械臂动作耦合。** 8 条独立 B90 示范、32 个中间状态、10 个成对采样种子：14→40 mm 的夹爪观测在第 10 个动作步使预测末端高度 **32/32 状态上升**，平均 **+20.95 mm**（范围 +3.28～+67.91 mm），状态内种子 std 平均仅 **1.56 mm**。
- **方向在动作块前 20 步高度一致，但幅度强烈依赖状态。** step0–19 全部 32/32 为正；早期状态平均 +55.2 mm，接近/推动状态约 +8～11 mm。到 chunk 末端（step49）只有 22/32 为正，出现 10 个负值，说明动作块后段可能反转。
- **14→25、25→35、35→40 不是同一效应。** step0 的 14→25 只有 12/32 为正（均值 +0.24 mm），而 25→35 与 35→40 都是 32/32 正（+11.3、+11.4 mm）；step10 的 14→40 全部为正。没有观察到 35→40 的特殊反向突变。
- **D2 B90 专家数据中，全局夹爪—垂直速度相关性很弱**：7,016 帧 / 60 条 yaw90，Pearson **0.110**、Spearman **−0.065**。夹爪开度与 progress 强混杂：宽夹爪几乎只出现在轨迹早期。
- **局部联合覆盖有限，且 40 mm 完全缺失。** 全局夹爪最大 **35.0 mm**，40 mm 帧 **0**。在 q≤0.30 rad 邻域内，32 个状态里只有 5 个具备“14 mm 与 35 mm 都有 ≥5 帧”的共同支持；q≤0.75 时有 9 个。
- **敏感性与局部覆盖有强描述性相关，但被状态/高度混杂。** 14→40 敏感性与邻域总帧数相关 **−0.73**，与 near-35 mm 帧数相关 **+0.92**；早期状态既有 35 mm 训练起点，又有最大敏感性。不能据此证明覆盖导致敏感性。

## Experimental Setup

| 项 | 值 |
|---|---|
| checkpoint | `outputs/smolvla_e23_D2/checkpoints/020000/pretrained_model` |
| sha256 | `34a6e3080a88250302a9f0abbc01608b58097324af2c321b00b84c395fbdd58b` |
| Git 基线 | `0e85c29` + 本实验工作区 |
| 任务 | `push the orange block along the positive X direction to the wall` |
| 数据 | `data/_shards_e23/B_D2`，B90 yaw90 60 条 / 7,016 帧（yaw0 为补充，未混入主统计） |
| 状态 | 8 条 yaw90 示范 × frames 10/25/39/55 = **32 个中间状态** |
| 反事实 | 14/20/25/30/35/40 mm 单指观测，**10 个成对种子/状态** |
| 动作块 | 50 步；FK 只用于预测目标轨迹 |
| 邻域 | q 距离 0.15/0.30/0.50/0.75 rad |
| 输出 | `results/experiment_2_8/`（`outputs/exp28_gripper_arm_coupling/` 为镜像） |

状态来源：实验 2.6 的 8 条 yaw90 轨迹（ep66,74,98,80,88,104,64,82），通过原始动作 replay 重建 Bullet，q replay 误差 ≤4.7×10⁻⁶ rad。阶段标签按 episode progress 分层（early<0.15, approach<0.3, precontact<0.4, push≥0.4），**不是**真实接触阶段；逐帧物体真值缺失。

## Verification and Reproducibility

- 夹爪观测确实是 `observation.state[7:9]` 两根手指的实际关节位置；14/40 mm 指**单指位置**，不是总开口。
- 模型动作是反归一化后的 7 关节绝对位置目标 + 夹爪目标；FK 使用 Panda link 11，世界坐标 +Z 向上。
- action chunk 长度 50；只分析预测目标，不把 FK 目标轨迹当成真实物理轨迹。
- D2 B90 数据只有 60 条 yaw90；用 collection 顺序 `ok[i]` 映射（yaw90 为 i=60..119），避免 HOME 重复起点导致 group 误配。
- 反事实图像来自每个状态的真实观测；**只修改本体观测**，图像与修改后夹爪数值可能不一致，属于 `Observation-only counterfactual intervention`。

## Results A — Frozen Policy Sensitivity

### 跨状态总体

32 个非 C3 状态，step10 的 14→40 mm：

- 平均值 **+20.95 mm**；最小 **+3.28 mm**；最大 **+67.91 mm**。
- **32/32 为正**；没有负向或近零状态。
- 状态内 10 个种子的 std 平均 **1.56 mm**，远小于跨状态差异。

按动作步：

| chunk step | 平均 Δz (mm) | 最小值 | 最大值 | 正状态数 |
|---:|---:|---:|---:|---:|
| 0 | 22.91 | 7.79 | 42.13 | 32/32 |
| 4 | 22.95 | 5.60 | 55.92 | 32/32 |
| 9 | 20.95 | 3.28 | 67.91 | 32/32 |
| 19 | 15.53 | 1.09 | 79.56 | 32/32 |
| 49 | 3.32 | −13.34 | 26.03 | 22/32 |

**RQ1：** 32/32 状态、8/8 条独立 B90 示范存在耦合。
**RQ2：** 前 20 步方向一致为正；幅度 3–68 mm；没有负向状态，chunk 末端出现 10 个负值。
**RQ3：** 强状态依赖：early 平均 +55.2 mm，approach +10.6，precontact +7.9，push +9.6。敏感性与初始 EE 高度相关 **+0.921**，与初始夹爪 +0.886，与 progress −0.690。

### 分区间

step10：

| 区间 | 平均 Δz (mm) | 正状态比例 |
|---|---:|---:|
| 14→25 | 7.09 | 20/32 |
| 25→35 | 8.66 | 30/32 |
| 35→40 | 5.20 | 31/32 |
| 14→40 | 20.95 | 32/32 |

step0 的 14→25 只有 12/32 为正（均值 +0.24 mm），说明最先 25 mm 的效应在部分状态可正可负；25→35 和 35→40 在 step0 都是 32/32 正。**RQ4：非线性；14→25 最弱且方向不稳定，25→35/35→40 更强；没有 35→40 的反向突变。**

![Figure 1](../results/experiment_2_8/figures/fig01_height_curves.png)
![Figure 2](../results/experiment_2_8/figures/fig02_sensitivity_heatmap.png)
![Figure 3](../results/experiment_2_8/figures/fig03_sensitivity_by_stage.png)

## Results B — Expert Training Dynamics

7,016 个有效 yaw90 时间步 / 60 条示范；垂直速度由相邻真实 q 的 FK 高度差 / 真实时间戳计算，单位 mm/s。

- 全局 vz：均值 **−64.8 mm/s**，std 130.8；负值表示下降。
- 夹爪：均值 14.7 mm，最大 35.0 mm，40 mm 帧 0。
- **Pearson(g, vz)=0.110；Spearman=−0.065**：全局相关性很弱且方向不一致。

按 progress：

| 分层 | 帧数 | 夹爪均值 | vz 均值 | vz std |
|---|---:|---:|---:|---:|
| 0–15% | 1,069 | 18.78 | −141.9 | 126.6 |
| 15–30% | 1,053 | 14.0 | −249.2 | 78.1 |
| 30–50% | 1,377 | 14.0 | −73.8 | 105.8 |
| 50–100% | 3,517 | 14.0 | +17.4 | 65.4 |

按夹爪分箱：

| 夹爪 | 帧数 | 独立 episode | vz 均值 |
|---|---:|---:|---:|
| <16 | 6,583 | 60 | −69.1 |
| 16–25 | 193 | 60 | +1.2 |
| 25–35 | 192 | 60 | −0.01 |
| ≥35 | 48 | 48 | 0.0 |

**RQ5：** 全局相关性弱；按 progress 分层后阶段本身解释大部分运动；宽夹爪只出现在早期，导致 **无共同覆盖的后期状态无法估计 14→40 的专家效应**。这不是“夹爪不影响专家”，而是数据不支持条件估计。

![Figure 4](../results/experiment_2_8/figures/fig04_expert_gripper_vz.png)
![Figure 5](../results/experiment_2_8/figures/fig05_gripper_by_stage.png)

## Results C — Joint-State Coverage

- 全局：夹爪 40 mm **0 帧**；35 mm 48 帧（48 条 episode 的起点）。
- q≤0.30 rad 邻域：中位 **1,457 帧 / 53 episodes**；**只有 5/32 状态**具备 14 mm 与 35 mm 的 ≥5 帧共同支持；仅 8/32 状态有任意 near-35 帧。
- q≤0.75 rad：中位 3,879 帧 / 60 episodes；9/32 状态有共同支持；29/32 有 near-35。
- 后期/推动状态的局部夹爪最大通常只有 14.0 mm；早期状态才出现 35 mm 起点。

**RQ6：** 全局“有 35 mm”不等于局部联合覆盖；大多数非 C3 状态没有“同一 q 附近既有 14 mm 又有 35 mm”的共同支持；40 mm 在任何邻域都没有。

![Figure 6](../results/experiment_2_8/figures/fig06_coverage_distribution.png)

## Results D — Cross Analysis

| 指标 | 相关性 |
|---|---:|
| 14→40 敏感性与 q≤0.30 总帧数 | **−0.731** |
| 14→40 敏感性与 q≤0.30 near-35 帧数 | **+0.924** |
| 敏感性与初始 EE 高度 | +0.921 |
| 敏感性与 progress | −0.690 |

- 早期状态（例如 ep082 f10）敏感性 +67.9 mm，局部帧数仅 89，但 near-35=6；晚期/推动状态 local 帧数 >1,500，但 near-35=0，敏感性约 +4～10 mm。
- 这形成“高敏感 ↔ 有 35 mm 覆盖、但总邻域帧少”的混杂；不能把 near-35 覆盖解释为敏感性的原因，因为早期状态同时有高 EE 高度和不同 q。
- 没有“敏感性强且 35/40 覆盖充分”的状态：40 mm 覆盖为 0，35 mm 覆盖也只在少数早期邻域。

![Figure 7](../results/experiment_2_8/figures/fig07_coverage_vs_sensitivity.png)

**RQ7：** 描述性相关存在，但被阶段/EE 高度混杂；正例是早期状态，反例是晚期低敏感状态；无法做因果解释。

## Limitations

- 反事实只改本体观测，图像仍是原状态；视觉—本体不一致，不能当作真实闭环输入。
- 只有 32 个状态、8 条轨迹、10 个种子；种子是策略采样，不是独立状态。
- 阶段标签是 progress 分层，不是接触真值；逐帧物体姿态缺失。
- FK 高度是动作目标的几何预测，不是实际执行轨迹。
- 训练数据中夹爪与阶段强混杂；局部 40 mm 覆盖为零，无法估计 40 mm 专家动作。
- 相关不等于因果；没有 attention/embedding 证据。
- 本实验不是 Direct A90→B90 组合评测。

## Research Questions — Direct Answers

| RQ | Answer | Evidence | Confidence / Limitation |
|---|---|---|---|
| RQ1 | **Yes** | 32/32 states, 8/8 episodes, 14→40 step10 positive | 强；仅这些状态 |
| RQ2 | **Mixed by step; positive early** | step0–19 32/32 positive, mean 3–68 mm; step49 22/32 | 强；后段反转 |
| RQ3 | **Yes, state-dependent** | early +55.2 vs push +9.6; corr EE +0.92 | 中；高度/阶段混杂 |
| RQ4 | **Nonlinear** | 14–25 weak/mixed at step0; 25–35/35–40 strong | 中 |
| RQ5 | **Weak global; conditional not estimable** | Pearson 0.11, Spearman −0.065; wide grip only early | 强（缺失覆盖） |
| RQ6 | **Limited local joint coverage; zero 40 mm** | 5/32 common support @0.30; 40 mm 0 frames | 强 |
| RQ7 | **Descriptive only** | corr near35 +0.92, frames −0.73, confounded | 弱因果 |
| RQ8 | **Scope limited** | 32 non-C3 states, 8 episodes, offline counterfactual | 不可外推到 Direct |

## Conclusions

**A. Directly observed facts**
- 14→40 mm 在前 20 个动作步使所有 32 个状态预测更高，平均 +20.95 mm。
- 敏感性在早期状态最大（+55 mm），接近/推动状态较小（+8–11 mm）。
- D2 训练数据 40 mm 帧为 0；局部共同覆盖有限。
- 专家全局夹爪—vz 相关很弱。

**B. Plausible interpretations（未验证）**
- 模型可能把“夹爪张开”与“较早/较高接近阶段”关联；早期状态敏感性最大与训练中宽夹爪只出现在早期一致。
- 35→40 不是统一的临界点，但 14→25 的弱效应说明响应非线性。
- 敏感性可能部分由初始 EE 高度/构型驱动，而不仅是夹爪数值。

**C. Unresolved**
- 视觉—本体不一致是否改变真实闭环行为。
- 40 mm 联合状态完全缺失时，补数据是否能消除敏感性/失败。
- 真实接触阶段与 progress 的对应关系。
- 这些耦合是否是 Direct A90→B90 失败的原因。

## Artifacts

```text
results/experiment_2_8/
  config.yaml
  run_manifest.json
  README.md
  report.md
  data/
    state_manifest.csv
    policy_predictions.csv        # server (large)
    policy_sensitivity.csv
    expert_dynamics.csv
    expert_distribution_summary.csv
    coverage_neighbors.csv
    coverage_summary.csv
    cross_state_analysis.csv
  policy/*.npz                    # per-state chunks/z (server)
  figures/fig01..fig07
  logs/
```

`outputs/exp28_gripper_arm_coupling/` 为同一结果的镜像路径。
