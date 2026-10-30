# 实验 2.9：物理一致性验证与低位联合状态定向干预（A/B）

完整报告：[ab/report.md](ab/report.md)  
配置：[ab/config.yaml](ab/config.yaml)  
运行清单：[ab/run_manifest.json](ab/run_manifest.json)

## 结果摘要

### A：物理一致输入

- 10 个新状态（5 Low /5 High），新种子 92,000,000+。
- 40/40 个宽度配置物理有效；最大物体位移 1.4×10⁻⁶ mm。
- 物理一致 14→40 mm，step10 高度差：
  - Low：**+61.7 mm**（5/5 正）
  - High：**+51.3 mm**（5/5 正）
- 14→35 是主要效应（+55.5 low / +48.8 high）；35→40 仅 **+6.2/+2.5 mm**。
- 固定本体、只改图像夹爪：效应 **<1 mm**；本体观测主导。

### B：专家—D2 配对

4 个状态 ×3 宽度：

| 条件 | 成功 |
|---|---:|
| E0 原始专家 | **12/12** |
| E1 状态感知专家 | **12/12** |
| 冻结 D2 | **17/24** |

D2 按宽度：14 mm **6/8**、35 mm **4/8**、40 mm **7/8**。

- `low_00` +35 mm：专家成功，D2 **0/2**；但 `low_03` 为 1/2、`high_00` 为 2/2。
- 7 个 D2 失败：5 次 tilt>12°，1 次侧向 dy 32.4 mm，1 次 yaw 变化约 16.2°。
- 首次接触多为 link10，法向 z 分量小；**不是 C3 的纯 −Z 顶部接触**。
- 40 mm 并不比 35 mm 差（7/8 vs 4/8）；状态依赖明显，2 个 D2 采样/格。

## C 阶段：No-Go

没有进入 C，没有训练、没有新增数据、没有修改 checkpoint。

理由：

- 只有 `low_00 +35` 出现专家成功、D2 0/2 的单状态缺口；
- 35 mm 在 `high_00` 为 2/2，`low_03` 为 1/2；
- 40 mm 成功率反而更高；
- 失败机制是 tilt/yaw/侧向，不是统一顶部接触；
- 每格仅 2 个 D2 采样，不能稳定估计状态级成功率。

## 文件

- A/B 数据：`ab/data/`
- 图：`ab/figures/figA1–A3`、`figB1–B3`
- 视频（服务器）：`/scratch/lijh/simulate_vla/results/experiment_2_9/ab/videos/`
- 代码：
  - `scripts/exp29_prepare_states.py`
  - `scripts/exp29_modality.py`
  - `scripts/run_b90_e1.py`
  - `scripts/exp29_analyze_b.py`
  - `scripts/exp29_visualize_a.py`
  - `scripts/exp29_visualize_b.py`

## B2 补充：真实 Low-Near（结果摘要）

完整报告：[b2_low_near/report.md](b2_low_near/report.md)

- 原 `low_00` 的 14/35 **并非轨迹接近**：z@0.5s 114.7 vs160.5 mm；z@1.0s 18.4 vs70.3 mm；接触高度 17.5 vs16.3 mm，但接触时间 1.18 vs1.68 s，路径 0.209 vs0.259 m。差异在接近段持续，仅接触高度附近收敛。
- 新 3 个真正 Low-Near（EE z 0.070–0.085，后方0.07–0.09m）：14→40 预测仅 **+9.9 mm**，低于旧 Low +61.7、Mid-Near +62.2。
- D2 物理 30 次 **25/30**；Low-Near **15/18**（00/01 均6/6，02为3/6）；14/35/40 = **8/10、8/10、9/10**。
- E0 Low-Near 7/9；E1 Low-Near 仅 **6/18**，低位分支不可靠。
- D2 5 个失败：3 tilt、2 侧向；无纯 −Z 顶部接触。
- **No-Go**，不训练。
