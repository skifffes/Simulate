# D2–C3 coverage analysis (corrected observation/action alignment)

## 关键修正

上一版把 C3 第一次 B **动作目标** 0.01656 m 与 D2 ep66 f39 **观测** 0.014 m 比较，得到 2.56 mm。修正后：

- C3 第一次 B **观测**夹爪 = **0.040 m**；
- 与 ep66 f39 观测差 = **26.0 mm**；
- 与 D2 B 起点观测（0.035 m）差 = **5.0 mm**；
- 2.56 mm 仅是 action-target vs action-target 的差。

## 结果

- 120 条 B 示范 / 14,653 帧（`data/_shards_e23/B_D2`）。
- 起点最近：q **0.9079–0.9136 rad**，ep74（yaw90, diverse）；夹爪差 5 mm，但 EE 差 152 mm、相对几何差 150 mm。
- interior 最近（全部）：q **0.1162–0.1197 rad**，ep22 f39（yaw0）。
- interior task-compatible（yaw90）：q **0.1710–0.1728 rad**，ep66 f39；EE 差 7.7 mm、相对几何约 1.9 mm、**夹爪差 26 mm**。
- 夹爪 ≤5.1 mm 的 interior 帧：**0**；≤10 mm 的 interior 帧只有每条示范第 1 帧（0.032 m），q 仍 0.9136 rad。
- **没有完整观测近邻**：关节近的帧夹爪差 26 mm；夹爪近的起点关节/EE 远。

## 文件

- `c3_query_states.csv`（观测状态）
- `d2_b_state_manifest.csv`
- `initial_nn.csv`、`initial_nn_yaw90.csv`（含 `grip_obs`/`grip_diff_mm`）
- `interior_nn.csv`、`interior_nn_yaw90.csv`（含夹爪观测）
- `gripper_constrained_nn.csv`（2/5.1/10 mm，starts/interior/yaw90/all）
- `all_frame_nn.csv`、`episode_deduplicated_nn.csv`
- `reference_distance_distribution.csv`
- `nearest_trajectory_comparison.csv`、`..._yaw90.csv`
- `analysis_summary.json`（含 `state_alignment_audit`）
- `figures/fig1..fig7`、`matched_frames/`

## 缺失

- D2 parquet 没有逐帧物体位姿/接触标签；ep66 f39 的相对几何用起点物体位置近似。
- C3 的 256×256 VLA 输入未落盘；只有 640×480 诊断图。
- 3 个 C3 seed；task-compatible 轨迹只有 ep66 一条。

详见 `docs/experiment_2_5_d2_coverage.md`。
