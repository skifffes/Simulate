# D2–C3 coverage analysis (partial)

本目录只包含 **initial-state nearest neighbor** 分析结果。

## 可用

- `c3_query_states.csv`：C3 seed 40000000 的 B 启动状态。
- `d2_b_state_manifest.csv`：120 条成功 B 示范的 `start_q`/`start_ee`/物体场景。
- `initial_nn.csv`、`initial_nn_yaw90.csv`、`initial_nn_by_relative_geometry.csv`、`initial_nn_by_ee_distance.csv`
- `reference_distance_distribution.csv`
- `analysis_summary.json`、`analysis_config.json`
- `figures/`：Figure 1/3/4/6

## 不可用（服务器网络不可达）

- D2 全帧训练数据集（`data/lerobot_panda_e23_D2`）无法读取，因此：
  - `interior_nn.csv`
  - `all_frame_nn.csv`
  - `episode_deduplicated_nn.csv`
  - `nearest_trajectory_comparison.csv`
  - Figure 2/5、matched frames
  均未生成。没有用 `D2_episodes.jsonl` 的起点或配置伪造中间帧。
- C3 seed 40000001/40000002 的查询状态缺失；本地只有 seed 40000000 的 trace。

## 关键数字

- C3 → D2 B 起点最小 q L2：**0.9136 rad**，episode 74（yaw90，diverse/task_relevant）。
- D2 起点内部最近邻（排除自身）：median 0.0、p95 0.292、max 0.430 rad；C3 0.914 **超出全体起点内部最大距离**。
- 该最近起点的物理差：EE 152.4 mm、相对 EE–物体 149.5 mm、姿态 0.0°、物体 yaw 2.2°、墙间隙 3.3 mm。
- 若按 EE/相对几何找最近（episode 38），距离 85 mm，但它是 yaw0、yaw 差 87°——几何条件不匹配。

详见 `docs/experiment_2_5_d2_coverage.md`。
