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
