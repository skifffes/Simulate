# 实验 4.5 最终报告：Action Expert-only 全参数微调 vs LoRA

日期 2026-11-06；基准提交 `060dadf`；OpenPI `215abfb`。
训练 07:39–10:47（12,000 updates），评测 10:52–11:45。

**核心结论：同样的 311 条专家示范与 12,000 updates 下，完整更新 Action Expert 并没有比 LoRA
更可靠地完成 A90/B90；在训练初态上 LoRA 反而明显更好（B90 成功 1/12 vs 0/12、推进中位
0.170 m vs 0.014 m）。** 因此本轮假设**不被支持**，不标记 ATOMIC_SKILL_LEARNED。

## 1. 十一个问题

**1. Action Expert-only 是否真正冻结了视觉语言主干？**
是。`parameter_partition.json` + `gate2_audit.json`：总计 51 个张量，冻结 32 个
（`PaliGemma/img` 22 + `llm` 非 `_1` 10，共 2923.3 M），6 步真实更新后**冻结张量 0/32 改变**；
optimizer state 只有 40 个叶、860,196,930 元素 = 恰好 2×430,098,464（Adam μ/ν）。

**2. 哪些参数完整参与更新？参数数量是多少？**
19 个张量、**430,098,464（430.1 M，12.83%）**：
Action Expert 的 `layers/mlp_1`、`layers/attn/{q,kv,attn_vec}_einsum_1`、
`pre_attention_norm_1/Dense_0`、`pre_ffw_norm_1/Dense_0`、`final_norm_1/Dense_0`，
加 `action_in_proj`、`action_out_proj`、`time_mlp_in`、`time_mlp_out`。
6 步后**19/19 全部改变**，不存在"只更新投影层"的情况。

**3. A40 上是否能够稳定训练？**
是。batch 4、峰值显存 **34,375 MiB**、JIT 247 s（冷）/16.6 s（热）、
吞吐 **1.05–1.10 step/s**、loss 无 NaN（第 1 步 1.1620 → 第 12000 步 0.0004）。

**4. 实际训练多少 optimizer updates、多少名义 epoch？**
**12,000 updates**，311 条 / 12,967 窗口，batch 4 → **3.70 名义 epoch**（与 4.4 Group B 相同）。
外加一个**明确标记的救援实验**：0.3×LR、4,000 updates（见 §5）。

**5. Action Expert 与 LoRA 的离线结果差异？**

| 指标（12,000 步） | AE | LoRA（4.4 B） |
| --- | ---: | ---: |
| 总体 joint MAE | 0.0070 | 0.0071 |
| A90 抓取 MAE | 0.0071 | **0.0069** |
| A90 Δz | +0.0004 | +0.0001 |
| B90 推动 cos | 0.636 | **0.826** |
| B90 Δz | **−0.0005** | +0.0024 |
| 采样方差（seed std） | **0.0011** | 0.0016 |

→ 互有胜负：AE 在 B90 Δz 与采样稳定性略优，LoRA 在 B90 推动方向余弦明显更好，总体 MAE 持平。

**6. 相同初始条件下 A90/B90 严格成功率？**

| 评测集 | AE | LoRA |
| --- | --- | --- |
| 冻结 val（16 条 × 3 seeds = 48） | **0/48** | **0/48** |
| 训练初态（6 条 × 3 seeds = 18） | **0/18** | **1/18** |

- val A90：AE 30 条全部失败（抓取 26、两指同时接触 3、抬起后 1）；LoRA 30 条全部失败
  （抓取 18、放置位置 7、抬起后 4、两指 1）；
- val B90：两者 **18/18 完全没有机器人-方块接触**（advance ≈ 0、终端距墙 0.183 m 未变）；
  val 是留出条件（物体 y=0.27），两边都不会外推；
- 训练初态 B90：AE 0/12（推进中位 0.014 m、距墙 0.161 m），LoRA **1/12**（推进中位 0.170 m、距墙 0.0065 m）；
- 训练初态 A90：两边 0/6，抓取仍是共同瓶颈。

**7. Action Expert 是否降低了接触阶段动作不稳定性？**
部分降低：B90 Δz 均值 −0.0005（LoRA +0.0024）、seed std 0.0011（LoRA 0.0016）；
但训练本身出现退化 checkpoint（6000 步 MAE 0.0219、B90 cos 0.032；8000 步 cos −0.058），
说明 dense 专家在该 LR 下**优化稳定性更差**，而不是更好。

**8. 是否解决了 A90 抓取失败与 B90 推动不足？**
没有。A90 抓取：AE 6 条训练初态仅 1 条抬起（lift 均值 0.0018 m）；
B90 推动：AE 中位 0.014 m，远低于 LoRA 的 0.170 m，也没有任何墙接触。

**9. 是否出现新的严重失败模式？**
出现一次 **tilt 60.5°（把方块推倒）** 的灾难样本（LoRA 最大 4.6°）；
另外 AE 的 A90 出现 3/30 次 `two_fingers_simultaneous`（两指同时接触但未形成夹持）。

**10. 现有证据是否足以支持"LoRA 容量是主要瓶颈"？**
**不足以支持。** 若 LoRA 容量是瓶颈，AE（参数范围更大、可训练参数相近但无低秩约束）
应在闭环更好；实际是离线持平、闭环更差。剩余瓶颈更可能是数据覆盖
（val 条件完全没有示范）、动作表示/监督目标，或闭环误差累积。

**11. 是否已经拥有可用于零样本 A→B 研究的可靠统一原子策略？**
**没有。** 两个模型在 val 上 0/48，训练初态最好也只有 LoRA 的 1/18。

## 2. 受控条件与公平性说明

- 相同基座 `pi05_droid`、相同 OpenPI、相同 311 条数据与 split、相同 Anchor-16/fixed、
  相同 DROID normalization、相同 batch 4、相同 12,000 updates、相同 seed；
- **相同 updates ≠ 相同 epoch**：LoRA 与 AE 都是 3.70 epoch（同一数据同一 batch），这条对本轮不构成混淆；
- **相同超参未必是 AE 的最优超参**：本轮按要求只改微调范围；发现退化后另做了明确标记的
  0.3×LR 救援实验（§5），不把它当作严格对照。

## 3. 产物

| 文件 | 内容 |
| --- | --- |
| `parameter_partition.json` | 51 张量的分区与参数量 |
| `weight_loading_audit.md` | Gate 1（含 adaRMS 0→0.00968 证据） |
| `smoke_test.md` | Gate 2/3/4（冻结 0/32、专家 19/19、34.4 GB、逐位一致续训） |
| `memory_benchmark.json` | 显存/吞吐/JIT |
| `checkpoint_manifest.json` | 7 个 AE ckpt + 2 个 opt state + LoRA 参考的 SHA256 |
| `checkpoint_selection.md` / `selection.json` | 联合规则与结果（两边都选 step 12000） |
| `offline_eval_results.json` / `offline_eval.md` | 逐 checkpoint 离线指标 |
| `paired_rollout_results.csv` | val 96 条配对闭环 |
| `paired_traininit_results.csv` | 训练初态 36 条配对闭环 |
| `failure_analysis.md` | 失败模式与证据 |
| `artifact_manifest.json` | 服务器路径与数据哈希 |
