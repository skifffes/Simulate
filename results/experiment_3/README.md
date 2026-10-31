# 实验 3：真实 A90 终态 → 数据集 B90 起点的过渡规划与可组合性 Pilot

- 报告：[report.md](report.md)
- 配置：[config.yaml](config.yaml)
- 来源：[provenance.json](provenance.json)

## 关键结果

- 5 个 A90 成功终态快照 **q/object 完全相同**，实际只有 **1 个唯一 A 终态**。
- B90 yaw90 真实起点 **60 个**；在 20 mm/10° 宽松场景阈值下 **60/60 兼容**，但物体位置差 **1.5–17.9 mm**，没有精确场景匹配。
- Pilot 选择 3 个 B 起点；3/3 目标静态有效、3/3 规划成功（碰撞检查的关节直线；RRT 备用）、3/3 执行完成。
- 执行 q 误差 **5×10⁻⁶–8×10⁻⁶ rad**，EE 误差 **0.0016–0.0037 mm**，物体漂移 **~10⁻⁸ mm**。
- 视觉：top MAE **0.058–0.574**，wrist MAE **0.0002–3.21**（ep074 物体偏 14.9 mm，腕部差异最大）。
- D2 B90：AtoB **3/9**，B 参考 **6/9**；逐对 ep074 0/3 vs1/3、ep0641/3 vs2/3、ep0702/3 vs3/3。
- **机器人状态匹配，但物体场景未精确匹配**；D2 差异不能单独归因于 transition。
- 能力边界：1 A ×60 近似兼容 B ×3 pilot → 3 规划/执行/机器人匹配 → 2/3 配对有 B90 成功；0 精确物体匹配。

## 数据

| 文件 | 内容 |
|---|---|
| `data/a_terminal_states.csv` | 5 个 A90 终态（1 unique） |
| `data/b_training_start_states.csv` | 60 个真实 B90 起点 |
| `data/scene_compatibility.csv` | 场景兼容汇总 |
| `data/candidate_pairs.csv` | A×B 候选表（60） |
| `data/planning_results.csv` | 3 个 pilot 规划 |
| `data/execution_tracking.csv` | 324 步实际执行 |
| `data/endpoint_matching.csv` | q/EE/物体/视觉误差 |
| `data/visual_matching.csv` | top/wrist MAE |
| `data/b_rollout_results.csv` | 18 次 D2 继承 |
| `data/composability_funnel.csv` | 分层统计 |

代码：`scripts/exp3_a90_terminals.py`、`exp3_b_starts.py`、`exp3_pilot.py`、`exp3_d2_prepare.py`、`exp3_funnel.py`、`exp3_visualize.py`、`server/run_e3_d2.sh`
