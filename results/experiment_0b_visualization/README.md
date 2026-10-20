# 实验 0B 可视化回放

本目录是 `results/experiment_0b/multistart_results.csv` 的回放，不重新求解 IK、不搜索参数、不覆盖 0B 数据。
每条回放按记录重建：先执行技能 A（z1/z2），重置组再回到 HOME，然后命令 CSV 中保存的 IK 候选，使用原电机控制器执行。

运行：`& ".\\.conda312\\python.exe" ".\\scripts\\visualize_experiment_0b.py"`，默认每 12 个仿真步渲染一帧（20 fps GIF）。

- 回放成功/失败与 CSV 是否全部一致：True
- 最大轨迹长度差：0.000e+00 rad
- 最大初始关节差：0.000e+00 rad

`overview_direct.png` 为四组直接交接的关键帧对照；`ee_paths.png` 为末端路径；各子目录含 `motion.gif`、逐帧 PNG 和该组 `result.png`。
`replay_metrics.csv` 保留来源记录编号、成功标志、行程、时间和误差，便于核对。

五组回放全部成功。此前 `z1_C_reset` 的 tracking_error 已定位为 `panda_link7` 与 `panda_hand` 的碰撞几何重叠：失败判定忽略该对、物理引擎仍求解，给第 5 关节附加约 12 N·m 制动；该对现已在动力学层关闭。可视化是确定性场景的复现，不是独立物理试验，也不是 VLA 泛化证据。
