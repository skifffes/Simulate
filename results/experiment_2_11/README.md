# 实验 2.11：物理一致专家条件下的 D2 动作偏差

- 报告：[report.md](report.md)
- 配置：[config.yaml](config.yaml)
- 来源：[provenance.json](provenance.json)

## 关键结果

- 5 条专家轨迹 / 386 个 20Hz 观测：ep66 f39 ×14/35，C3 seed40000000 ×14/35/40。
- C3 专家 3/3 成功；14/35/40 的动作目标相同（先闭合到14，再走 0.07→0.018 waypoint），专家条件差为 **0**。
- Teacher-forced D2（3 seeds）在初始观测上的 35−14 EE 目标差：
  - C3：h0 **+4.7**, h4 +17.8, h9 +17.2, h19 +16.9 mm；
  - ep66：h0 **+13.5**, h4 +23.3, h9 +10.7, h19 +12.1 mm；
  - C3 40−14：h0 +14.4, h19 +32.0 mm。
- 平均 q L2 h0：C3 0.068/0.068/0.074；ep66 0.041/0.044。
- C3 40 mm 初始目标偏高 **15.9 mm**，h19 偏高 **34.4 mm**；C3 14 mm 初始仅高1.5 mm。
- 专家不能为 35/40 提供“不同正确动作”；2.10 的 wide=0 是描述性缺口，不证明因果。
- 训练干预仅候选，No-Go。

## 数据

| 文件 | 内容 |
|---|---|
| `data/existing_expert_data_audit.csv` | 5 条专家记录审计 |
| `data/paired_initial_states.csv` | 2 个配对状态 |
| `data/expert_trajectories.parquet` | 386 帧专家轨迹 |
| `data/teacher_forced_observations.parquet` | 386 帧观测 |
| `data/teacher_forced_policy_actions.npz` | 5轨迹×3 seeds×chunk |
| `data/action_alignment_audit.csv` | action/q 对齐 |
| `data/policy_expert_action_error.csv` | 每帧/种子/horizon 误差 |
| `data/gripper_condition_response.csv` | 初始条件响应 |
| `data/condition_response_residual.csv` | 模型−专家条件差 |
| `data/phase_level_summary.csv` | 分阶段误差 |
| `data/c3_case_analysis.csv` | C3 14/35/40 初始对照 |
| `data/identifiability_summary.csv` | 可识别性 |

## 代码

`scripts/exp211_capture.py`、`exp211_teacher.py`、`exp211_meta.py`、`exp211_local.py`、`exp211_tables.py`、`exp211_visualize.py`、`exp211_visualize_extra.py`、`server/run_e211_c3expert.sh`、`run_e211_capture.sh`
