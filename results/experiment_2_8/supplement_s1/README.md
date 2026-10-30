# 实验 2.8-S1：D2 专家目标高度与夹爪状态

正式报告：[report.md](report.md)

## 数据

- `data/_shards_e23/B_D2`，B90 yaw90 **60 episodes / 7,016 frames**
- 40 mm 观测：**0 帧**；按数值容差归并，35–40 mm：**60 帧 / 60 episodes**
- 时间步 20 Hz；`action[0:7]` 为绝对关节目标，`action[7]` 为夹爪目标

## 关键结果

- 全局夹爪—`Δz_target`：Pearson **0.088**，Spearman **−0.031**；与 `vz_actual`：0.098 / −0.049。
- h0 分箱：14 mm `Δz_target` −3.46 mm（下降率31.9%）；16–25 −2.81（30.2%）；25–35 / 35 mm 0.00（0%）。
- 但 h9 的 wider 组为 −67.9/−44.4 mm（vs 14 mm −34.9）；h19 为 −193.0/−167.4（vs −65.6）；“宽夹爪保持”只在 h0 成立，未来 horizon 反转。
- 60/60 episode 闭合→目标下降为正，中位 **0.25 s**；实际下降中位 0.25 s；ep66 延迟 1.90 s、ep96 0.65 s。
- progress 0–15% 有四组共同覆盖（14/16–25/25–35/35）；15% 后几乎仅14 mm。
- q≤0.30 仅 **5/32** 状态有 14 mm+35 mm 共同支持；40 mm 为0；q≤0.75 为9/32。
- 32 状态模型敏感性与局部专家 `Δz_target` 相关 **−0.775**，与初始 EE 高度 **+0.921**。

## 文件

| 文件 | 内容 |
|---|---|
| `data/expert_target_dynamics.csv` | 7,016 帧：q、grip、z_obs、z_target、Δz_target、vz_actual |
| `data/expert_chunk_targets.csv` | frame×horizon 目标高度，含 valid mask |
| `data/gripper_target_statistics.csv` | 夹爪分箱统计 |
| `data/episode_timing_events.csv` | 每条 episode 的闭合/下降事件 |
| `data/timing_summary.csv` | 三套阈值的时间差汇总 |
| `data/conditional_target_statistics.csv` | progress×grip 条件统计 |
| `data/local_target_coverage.csv` | 32 状态×4 邻域 |
| `data/model_expert_comparison.csv` | 32 状态模型敏感性 vs 专家局部统计 |
| `figures/fig01–fig08` | 图 |

## 代码

- `scripts/exp28s1_target_height.py`
- `scripts/exp28s1_visualize.py`

```bash
# server
HF_HUB_OFFLINE=1 PYTHONPATH=src venv/bin/python scripts/exp28s1_target_height.py
# local figures
python scripts/exp28s1_visualize.py
```
