# Normalization 决策（Gate 3）

## 结论

**训练使用官方 `pi05_droid` 的 DROID quantile stats**（`assets/droid/norm_stats.json`，
md5 `2e8bc5547dc797f1df5f4e5401136082`），不做自定义归一化。
理由见下（前 7 维 clip 比例为 **0**，夹爪维的差异是阈值效应而非尺度错误）。

## 证据（只用 train split 统计，`results/experiment_4_2/normalization_stats_check.json`）

统计口径：`exp4.1-split-v1` 的 **train** 61 条 episode / 10 882 帧（val/test 未参与）。

| 维度组 | train q01 | train q99 | DROID q01 | DROID q99 | 超出 DROID q99 的比例 | 归一化后 clip 比例 |
| --- | --- | --- | --- | --- | --- | --- |
| state 关节 0–6 | −2.44 … 0.61 | −1.51 … 2.85 | −3.32 … −1.54 | 1.22 … 3.14 | 0 | **0.000** |
| state 夹爪 (第 8 维) | 0.348 | 1.000 | 0.0 | 0.9998 | 0.282 | 0.282 |
| action 关节 0–6 (Δq) | −0.042 … −0.013 | 0.013 … 0.049 | −0.458 … −0.646 | 0.448 … 0.734 | 0 | **0.000** |
| action 夹爪 (第 8 维) | 0.35 | 1.000 | 0.0 | 0.9998 | 0.310 | 0.310 |

## 解读

1. **关节维完全落在 DROID 统计范围内**（clip = 0）。我们的 Δq 明显更小（约 0.05 rad/step vs
   DROID 的 ~0.45–0.8），原因是专家移动慢、20 Hz 下每步位移小。这是**尺度差而不是符号/单位错误**：
   两侧都是 rad/control-step。
2. **夹爪维的 0.28–0.31 不是尺度错误**：DROID 的 q99 = 0.9998 而我们的张开值是 1.0，
   归一化后 1.0004 被 clip 到 1.0（差 0.0004），只因"等于上界"而被计数。
   闭合值 0.425 归一化后约 −0.15，完全在范围内。
3. 若改用 train-split 自算 stats（Δq 的 q01/q99 ≈ ±0.04），归一化后数值会被放大 ~10 倍，
   与 `pi05_droid` 预训练输出分布不一致，需要更长训练才能收敛——**不作为首选**。
   自算 stats 的数值已保存在 `normalization_stats_check.json`，供 4.3 需要时复核。

## 反归一化后的物理幅值检查

用 DROID stats 反归一化后，模型输出的物理 Δq 上限为 ±0.45 rad（第 1 维 DROID q99），
而我们的数据只用 ±0.05。因此：
- 训练早期模型可能输出偏大的 Δq（被 action adapter 的 `clip(Δq, ±1)` 与
  `max_delta_rad=0.25` 双重限制住），不会造成危险动作；
- 这一点必须在闭环 smoke 里实测，不能只靠推断（见 `atomic_eval_summary.json` 的动作幅值统计）。

## 版本化

训练 checkpoint 与 `norm_stats_md5` 一起记录在 `checkpoint_manifest.json`；
推理服务加载同一份 assets 目录，确保训练/推理归一化一致。

## 审计更正（2026-11-04）：跨度口径必须用真实 16 步锚点标签

上表的 action 关节行是**单步 Δq**（`A[t]-S[t]`），而 4.2 实际训练标签是
**固定锚点的 16 步累计偏移**（`A[t+k]-S[t]`，见 `train_lora.make_item`）。
两者只在 k=0 相等，所以"跨度仅为 DROID 的 2.4–6.1%"**不能**用来描述训练标签。

用审计脚本在真实 train 数据上重算（`contract_replay_p0c/action_contract_audit.json`，
61 episodes / 2521 windows）：

| 口径 | 关节跨度 / DROID 跨度 |
| --- | --- |
| 单步 Δq（旧口径） | 0.024–0.061 |
| **16 步锚点标签（真实训练口径）** | **0.175–0.761** |
| 归一化后超出 [−1, 1] 的比例 | **0.000** |
| Normalize→Unnormalize 往返误差 | **1.1e-16** |

逐 k 的锚点标签 `abs_mean`：k=0 **0.008** → k=5 0.042 → k=15 **0.105 rad**（自然随 k 增长）。
归一化后（k=0）abs_mean 0.036、std 0.094、q95 0.245、max 0.487；k=5 时 std 0.20、q95 0.46。

**结论修订**：在真实训练标签口径下，关节有效信号占 DROID 量程的 **17.5%–76%**，
且没有任何标签被 clip 到 ±1，往返误差 1e-16。
因此"归一化把有效动作信号压窄"从 `HYPOTHESIS` 降级为 **WEAKENED**：
不再作为下一次单变量训练的首选干预（见 `contract_replay_p0c/final_report.md` Q8）。
