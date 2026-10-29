# 实验 2.8：跨中间状态夹爪—机械臂耦合

正式报告：[report.md](report.md)

## 数据规模

- 8 条 yaw90 B90 轨迹 × frames 10/25/39/55 = **32 个中间状态**
- 每个状态 6 种夹爪观测（14/20/25/30/35/40 mm）× **10 个成对种子**
- 训练数据：60 条 yaw90 示范 / 7,016 帧
- 邻域阈值：0.15/0.30/0.50/0.75 rad

## 关键结果

- 14→40 mm 在 step10 使 **32/32** 状态预测更高，平均 **+20.95 mm**（+3.28～+67.91 mm），状态内种子 std 1.56 mm。
- 前 20 步 32/32 为正；step49 只有 22/32，出现 10 个负值。
- 早期状态平均 +55.2 mm，接近/推动状态 +8～11 mm；与初始 EE 高度相关 +0.92。
- 14→25 在 step0 只有 12/32 为正；25→35、35→40 都是 32/32 正。
- D2 训练夹爪最大 35 mm，40 mm 帧 0；全局专家夹爪—vz Pearson 0.11、Spearman −0.065。
- q≤0.30 邻域中位 1,457 帧 / 53 episodes，但只有 5/32 状态有 14 mm + 35 mm 共同支持；q≤0.75 为 9/32。

## 命令

```bash
# 服务器
venv/bin/python scripts/reconstruct_b_midstates.py --states 66:10,66:25,... --out results/experiment_2_8/states/reconstructed
venv/bin/python scripts/exp28_prepare_states.py
bash server/run_e28_policy.sh
venv/bin/python scripts/exp28_training_dynamics.py
venv/bin/python scripts/exp28_local_coverage.py
venv/bin/python scripts/exp28_joint_analysis.py
# 本地
python scripts/exp28_visualize.py
```
