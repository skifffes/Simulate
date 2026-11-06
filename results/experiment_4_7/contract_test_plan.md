# 实验 4.7 契约测试计划（执行前冻结）

日期 2026-11-06；基准提交 `a659809`；OpenPI `215abfb`。
本轮**不训练、不扩数据、不改超参**，只验证"训练 → 推理 → 物理执行"的数值一致性。

## 固定样本

| skill | episode | 说明 |
| --- | ---: | --- |
| A90 | 0 | 训练集，严格成功（4.6 用过） |
| A90 | 5 | 训练集，严格成功；4.6 中曾抬起 0.17 m |
| B90 | 0 | 训练集，严格成功 |
| B90 | 2 | 训练集，严格成功 |

数据根 `/scratch/lijh/exp46/export_overfit`（符号链接到 export_v3），
provenance 见 `results/experiment_4_6/overfit_subset.json`。
每个 episode 取 9 个帧：0、1/5、2/5、3/5、4/5、末尾-17、夹爪闭合前 1 帧、闭合帧、闭合后 5 帧。

冻结模型：
- exp4.6 LoRA step3000（`/scratch/lijh/exp46/run/ckpts/lora_step03000.npz` → 合并到 `/scratch/lijh/exp47/merged/step3000`）；
- exp4.4 Group B LoRA step12000（Gate 5 可选对照）。

## Gate 与容差（先定义，后测量）

| Gate | 指标 | 容差 | 判定 |
| --- | --- | --- | --- |
| 1 图像 | 同一原始帧经两路径后的张量 max abs | ≤ 1e-6（float32 归一化后） | 全部字段 PASS 才算 |
| 1 token/mask | token IDs、mask | **必须逐位相等** | 相等 |
| 1 state | 归一化后 state max abs | ≤ 1e-6 | — |
| 2 Anchor-16 | \|q_target − A[t+k]\| | ≤ 1e-6 rad（无裁剪时） | k0、k1–7、k8–15 分别统计 |
| 2 裁剪 | 触发 `max_delta_rad*(k+1)` / ±1 rad 拒绝的行 | 记录比例，不设阈值 | 报告 |
| 3 专家回放 | v1.2 严格成功 | 4/4 | 成功=专家成功 |
| 3 Oracle B1 | v1.2 严格成功 | 4/4 | 成功=chunk+Adapter+执行链正确 |
| 3 Oracle B2 | v1.2 严格成功 | 记录（允许因状态漂移失败） | 不作为 bug 判据 |
| 4 FK | pose() vs 同 link FK | ≤ 1e-3 m | 20 个构型 |
| 5 推理一致性 | 离线 vs Policy 的 16×8 动作 max abs | ≤ 5e-3 rad（bf16 + 两次 jit 的数值精度） | 8 个样本 |
