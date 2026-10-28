# D2–C3 coverage analysis (full joint-space NN)

## 结果

- 120 条 B 示范 / 14,653 帧（`data/_shards_e23/B_D2`）。
- C3 三个 seed 的 start 最近：**0.9079–0.9136 rad**，episode 74（yaw90，diverse）。
- interior 最近（全部）：**0.1162–0.1197 rad**，episode 22 f39（yaw0）。
- interior 最近（task-compatible yaw90）：**0.1710–0.1728 rad**，episode 66 f39（yaw90）。
- ep66 f39：EE 差 7.7 mm、相对 EE–物体差 **1.87 mm**、夹爪 14.0 vs 16.6 mm；专家下降，C3 VLA 前 6 tick 上升。
- yaw90 top-20 只来自 ep66 一条示范；全帧 top 还包含 yaw0 的 ep22/12/38 等。

## 文件

- `c3_query_states.csv`：3 个 C3 B 启动状态。
- `d2_b_state_manifest.csv`：120 条 B 起点/场景。
- `initial_nn.csv`、`initial_nn_yaw90.csv`
- `interior_nn.csv`、`interior_nn_yaw90.csv`
- `all_frame_nn.csv`、`episode_deduplicated_nn.csv`
- `reference_distance_distribution.csv`
- `nearest_trajectory_comparison.csv`、`..._yaw90.csv`
- `analysis_summary.json`、`analysis_config.json`
- `figures/`、`matched_frames/`

## 缺失

- 训练 parquet 没有逐帧物体位姿和接触标签；ep66 f39 的相对几何用起点物体位置近似（wrist 图显示尚未接触）。
- 只有 3 个 C3 seed；task-compatible 轨迹只有 1 条。
- 未做视觉特征最近邻（训练图像已抽帧，但未计算图像嵌入）。

详见 `docs/experiment_2_5_d2_coverage.md`。
