# 实验 3 受控诊断：A/B/C 启动状态等价性与冻结 D2

## 目标

在 **ep064 f000** 上控制 A/B/C 的启动状态，区分：

- **A**：B 参考场景（object_B）；
- **B**：A 场景 + 诊断性 qB 放置（object_A）；
- **C**：A 场景 + 真实规划过渡（object_A）。

先做状态对齐，再比较配对种子的冻结 D2 B90 行为。

## 关键结果

- B/C 启动状态在全部记录变量上通过：q 8.8e-7 rad、object 3.6e-8 mm、top/wrist MAE 0、contacts 0；**B/C 图像逐像素相同**。
- A/B 仅物体位置差 **1.864 mm**、yaw axis 2.54°、top MAE 0.058；q/EE/grip 相同。
- 5 个配对种子：**A 5/5，B 4/5，C 4/5**；B/C 同失败 seed4（tilt 15.1°/17.9°），A seed4 成功。
- **旧 AtoB 3/9 vs Bref 6/9 未复现**；旧 runner 未显式配对策略种子（A row seed 53050005 vs B/C 64）。
- **v2.2 成功 13/15，但 wall_contact 仅 1/15**；成功判据不是墙接触。
- 完整视频：`videos/AtoB_full.mp4`。

## 运行顺序

```bash
# 1) 构建 A/B/C 初始状态（B 为诊断 reset；C 使用已有 transition endpoint）
HF_HUB_OFFLINE=1 PYTHONPATH=src venv/bin/python scripts/exp3_controlled_states.py

# 2) 状态对齐与图
python scripts/exp3c_alignment.py
python scripts/exp3c_visualize.py

# 3) 配对种子 rollout（15 次；3 组 ×5 seeds）
bash server/run_e3c.sh
python scripts/exp3c_parse.py
python scripts/exp3c_summaries.py

# 4) 视频
python scripts/exp3c_video.py
python scripts/exp3c_join_video2.py
```

## 文件

| 路径 | 内容 |
|---|---|
| `initial_states/` | A/B/C bullet + JSON + 256 top/wrist |
| `data/initial_states.csv` | 完整启动状态 |
| `data/state_alignment.csv` | 逐变量差异与阈值 |
| `state_alignment.md` | 可读状态对齐表 |
| `data/b90_rollout_results.csv` | 15 次配对 rollout |
| `data/first_action.csv` | 首段动作 |
| `summaries/group_summary.csv` | 分组成功率/接触/倾斜 |
| `summaries/paired_seed.csv` | 逐 seed 对照 |
| `summaries/first_action_diff.csv` | 首动作 q L2 |
| `rollout/` | 原始 rollout（服务器，含 trace/video） |
| `videos/AtoB_full.mp4` | A 终态→过渡→B90 seed0 |
| `report.md` | 研究分析 |

## 重要限制

- 只有 1 个 B 目标、1 个唯一 A 终态、5 个配对 seed；不是统计泛化。
- B Reference 是重建场景，不是原始训练快照；RGB 也是重建。
- 控制器目标未保存；B/C 首动作仍有 0.0013–0.0022 rad 数值差，未完全分离。
- A/B 的物体差与 top 图像差同时存在，不能分离。
- 规划为碰撞采样直线，不是完整动态安全证明。
- v2.2 success 与 wall_contact 分开报告；14/15 成功无墙接触。
