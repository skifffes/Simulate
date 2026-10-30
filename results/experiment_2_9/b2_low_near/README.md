# 实验 2.9-B2：真实 Low-Near 闭环与接触

- 报告：[report.md](report.md)
- 数据：`data/`
- 图：`figures/fig01–fig09`
- 视频：服务器 `/scratch/lijh/simulate_vla/results/experiment_2_9/b2_low_near/videos/`

## 关键数值

- 原 low00：14 vs35，z@0.5s **114.7 vs160.5 mm**；z@1.0s **18.4 vs70.3 mm**；接触高度 17.5 vs16.3 mm，但接触时间 1.18 vs1.68 s。**早期不接近，仅接触点附近收敛。**
- 新状态：3 Low-Near（EE z 0.070–0.085，后方0.07–0.09m）、2 Low-Far、2 Mid-Near；21/21 宽度配置有效。
- 预测 14→40 step10：Low-Near **+9.9 mm**（3/3）、Low-Far +24.8、Mid-Near +62.2；旧 Low +61.7。
- D2 物理：30 次 **25/30**；Low-Near **15/18**；宽度 14 **8/10**、35 **8/10**、40 **9/10**。
- E0 Low-Near 7/9；E1 总 18/30，Low-Near 6/18；E1 低位分支触发但不可靠。
- D2 5 个失败：3 tilt、2 侧向；无纯 −Z 顶部接触。
- C 阶段 **No-Go**。

## 文件

| 文件 | 内容 |
|---|---|
| `data/original_low00_recheck.csv` | 旧 low00 接触/高度/路径 |
| `data/original_low00_trajectories.csv` | 旧 low00 20Hz 轨迹 |
| `data/state_manifest.csv` | 7 个新状态 |
| `data/state_physical_validity.csv` | 21/21 |
| `data/policy_predictions.csv` | 新状态 14/35/40 全 chunk |
| `data/policy_height_sensitivity.csv` | 预测敏感性 |
| `data/rollout_outcomes.csv` | 69 次物理执行 |
| `data/actual_ee_trajectories.csv` | 20Hz 实际轨迹 |
| `data/controller_targets.csv` | 发送给控制器的目标 |
| `data/expert_trajectories.csv` | E0/E1 |
| `data/contact_events.csv` | 首次接触 |
| `data/trajectory_pair_comparison.csv` | 14 vs35/40 配对 |
| `data/state_level_summary.csv` | 状态×宽度成功/预测 |

## 代码

`scripts/exp29b2_prepare_states.py`、`exp29b2_policy.py`、`exp29b2_recheck.py`、`exp29b2_validity.py`、`exp29b2_analyze.py`、`exp29b2_visualize.py`、`exp29b2_visualize_extra.py`、`server/run_e29b2.sh`
