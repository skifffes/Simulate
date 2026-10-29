# 实验 2.7：夹爪观测因果诊断

- 报告：[docs/experiment_2_7_gripper_causality.md](../../docs/experiment_2_7_gripper_causality.md)
- 汇总：[summary.json](summary.json)、[config.json](config.json)

## 目录

- `observation_intervention/`：固定 C3 图像/七关节，仅改夹爪本体观测 14/20/25/30/35/40 mm；20 个成对采样种子，两套真实图像（real40/real14）。`paired_action_chunks/` 在服务器。
- `crossed_action_replay/`：P14/P40 七关节动作 × 14/40 mm 初始夹爪的开放环交叉重放（4 条件），含 `contact_statistics.csv`。
- `d2_training_analysis/`：120 条 B90 / 14,653 帧的夹爪开度—阶段统计与 C3 局部覆盖。
- `figures/`：Fig1–Fig6。
- `videos/`：4 个对照视频（服务器）。
- `c3_ee_height.csv`：C3 14/40 mm 真实 EE 高度。

## 关键数字

- 反事实：夹爪 14→40 mm，七关节动作 L2 0→**0.141 rad**；预测 Δz first10 **−5.31→−3.48 cm**；20 种子的 std 仅 0.0044 rad / 0.78 cm。
- 交叉重放：P14 路径 + 14/40 mm 都成功（侧向接触 link10，法向≈水平）；P40 路径 + 14/40 mm 都失败（顶部接触 link9，法向≈−Z，gap≈17 cm）。
- C3 实际：14 mm 侧向接触（EE z 19.9 mm，成功）；40 mm 顶部接触（47.8 mm，tilt 78.1°，失败）。
- D2 训练：夹爪最大 35 mm，**40 mm 帧 0**；C3 附近 q≤0.75 的 584 帧全部为 12.75–14.10 mm。

## 限制

- 反事实状态与真实图像不一致，仅用于敏感性诊断。
- 交叉重放是 open-loop，不是冻结 VLA 的闭环成功率。
- 训练数据无逐帧物体真值；阶段是 progress/EE 高度估计。
- 只有 1 个 C3 状态、1 条 P14、1 条 P40 源轨迹。
