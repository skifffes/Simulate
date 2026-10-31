# 实验 2.10：D2 B90 数据中的夹爪—机械臂条件关系审计

- 报告：[report.md](report.md)
- 配置：[config.yaml](config.yaml)
- 来源：[provenance.json](provenance.json)

## 关键结果

- D2 全量 **480 集 / 68,892 帧**；B90 yaw90 **60 / 7,016**。
- B90 实际夹爪：Narrow 6,585 帧（close/descend/push），Intermediate 228 帧（全 initial），Wide 60 帧（全 initial）。
- wide 的 Δz_target：h0 0、h9 −44.4、h19 −167.4 mm；不是“宽=高”。
- C3 q 邻域在 r=0.75 有 126–147 帧/3–4 eps，但 **wide=0**；ep66 f39 1,731 帧/52 eps，**wide=0**。
- 43 个查询状态 r=0.30 common support **0/43**；r=0.5/0.75 的 wide 仅 5–7 个 initial 帧，阶段不匹配。
- 模型 14→35：32/32 正，均值 **+15.75 mm**；C3 **+16.96 mm**（14→40 +32.0）。
- 专家侧无 same-q wide 标签，模型响应与专家不可比较（Not Identifiable）。
- H1 Supported（时序）/Weakened（宽=高）；H2 缺口 Observed、因果 Not Identifiable；H3 Not Identifiable。

## 数据

| 文件 | 内容 |
|---|---|
| `data/dataset_audit.csv` | 480 集分任务统计 |
| `data/b90_frame_state_action.parquet` | 7,016 帧联合表 |
| `data/expert_action_horizons.parquet` | 5 horizons + valid mask |
| `data/expert_phase_events.csv` | per-episode close/descent/low |
| `data/global_gripper_action_stats.csv` | 3 分箱统计 |
| `data/queried_states.csv` | 43 状态 |
| `data/local_joint_coverage.csv` | 4 邻域覆盖 |
| `data/local_common_support.csv` | 共同支持 |
| `data/neighbor_expert_labels.csv` | top20 邻居标签 |
| `data/policy_response.csv` | 模型 14→35/40 |
| `data/supported_policy_expert_comparison.csv` | 13 行，全部 comparison_valid=False |
| `data/identifiability_matrix.csv` | H/Q 判断 |

## 代码

`scripts/exp210_build.py`、`exp210_coverage.py`、`exp210_local.py`、`exp210_visualize.py`
