# 实验 2.9 A/B 交付

- 完整报告：[report.md](report.md)
- 配置：[config.yaml](config.yaml)
- 运行清单：[run_manifest.json](run_manifest.json)

## 数据

| 文件 | 内容 |
|---|---|
| `data/state_manifest.csv` | 10 个 Low/High 新状态 |
| `data/physical_validity.csv` | 40 个宽度配置；40/40 有效 |
| `data/modality_interventions.csv` | 5 种图像/本体条件 ×10 状态 ×10 种子 |
| `data/policy_height_sensitivity.csv` | 14→35 / 35→40 / 14→40 敏感性 |
| `data/expert_rollouts.csv` | E0/E1 12+12 |
| `data/frozen_policy_rollouts.csv` | D2 24 |
| `data/expert_policy_comparison.csv` | 状态×宽度汇总 |
| `data/contact_analysis.csv` | 首次接触 |
| `data/failure_cases.csv` | 7 个 D2 失败 |

## 关键数值

- A：物理一致 14→40 step10：Low **+61.7 mm**，High **+51.3 mm**；35→40 仅 +6.2/+2.5 mm。
- A：本体主导；图像夹爪效应 <1 mm。
- B：E0 **12/12**、E1 **12/12**、D2 **17/24**；D2 14/35/40 = **6/8、4/8、7/8**。
- B：`low_00` +35 mm 为 **0/2**（专家成功）；`high_00` +35 mm 为 2/2。
- 失败：5 次 tilt、1 次侧向 dy、1 次 yaw；非法向纯 −Z 顶部接触。
- C：**No-Go**，未训练。

## 图/视频

- `figures/figA1–A3`、`figures/figB1–B3`
- 服务器视频：`/scratch/lijh/simulate_vla/results/experiment_2_9/ab/videos/`
  - `B_low00_g35_expert_vs_D2.mp4`
  - `B_low00_D2_g14_vs_g35.mp4`
  - `B_high03_g35_expert_vs_D2.mp4`
  - `B_low00_g40_expert_vs_D2.mp4`
