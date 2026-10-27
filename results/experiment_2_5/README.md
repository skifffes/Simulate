# 实验 2.5 结果目录

## 条件

- `G1/C0`：A90→B 原始硬切换（复用 2.4 G1 的 seed 40000000–40000002）
- `G1/C1`：延迟闭爪 0.5 s（执行开度钳制 ≥0.035 m）
- `G1/C2`：安全离开物体（+8 cm Z，−6 cm X）
- `G1/C3`：Oracle 预推动位姿（obj_x−0.10, obj_y, 0.07）
- `G1/C4`：连续退回 HOME
- `G5/R0`：2.4 原始局部退让（复用 2.4 G5 的 3 个 seed）
- `G5/R1`：clearance-first（+8 cm Z，−2 cm X）
- `G3`：B 独立 yaw90 原子参考（2.4，10 次）

## 关键结果（3 个配对 seed）

| 条件 | B 成功 | max lift 均值 |
|---|---:|---:|
| C0 | 0/3 | 9.8 cm |
| C1 | 0/3 | 9.0 cm |
| C2 | 0/3 | 2.5 cm |
| C3 | 0/3 | 0.8 cm |
| C4 | **3/3** | 0.33 cm |
| R0 | 0/3 | 0.06 cm |
| R1 | 0/3 | 0.95 cm |
| G3 | 9/10 | 0.58 cm |

## 文件

每个 `seed_*/` 含与 2.4 相同的日志/视频：`physics_trace.csv`、`contact_trace.csv`、`policy_trace.jsonl`、`success_timeline.csv`、`frame_index.csv`、`event_summary.json`、`metadata.json`、`top/side/contact/comparison/events.mp4`。

- `summary.csv` / `analysis.json`：本轮统计。
- `figures/`：fig1（C1 夹爪指令/实际）、fig3（C1 EE/物体 Z）；fig4/fig5 在服务器。
- 服务器路径：`/scratch/lijh/simulate_vla/results/experiment_2_5/`。

## 限制

- 每条件仅 3 个配对 seed。
- C2/C3/R1 的连接段 skill 标签已修复并重跑：C2/C3 连接 270/244 ticks、倾斜 0.15°；R1 163 ticks、倾斜 0.24°。本目录数据为修正后版本。
- G5 R0 连接段物体倾斜 23.9°、位移 7.5 mm；R1 仅 0.24° / 4.1 mm。R0 的 robot–object 接触计数仍为 0，单指扫掠碰撞的具体 link 需下一轮逐 link 接触检查确认。
