# Gate 5：模型推理一致性 + 端到端误差预算

## 5.1 离线路径 vs Policy 路径（`policy_inference_parity.csv`）

同一帧、同一 checkpoint（exp46 LoRA step3000，合并到 `/scratch/lijh/exp47/merged/step3000`）、
同一 RNG：

- 离线：训练路径输入 → `model.sample_actions(rng, obs)`；
- 在线：客户端 obs → `Policy.infer(obs)`（内部同样 `sample_actions`，但会**先 split 一次 rng**）。

**关键发现（自查）**：第一版对比用 `key(0)` 直接喂给离线 `sample_actions`，
与 `Policy.infer` 的 `split(key(0))[1]` 不是同一 key，导致 max abs **0.19 rad** 的假差异。
对齐 split 后：

| 指标 | 8 个样本的结果 |
| --- | ---: |
| 16×8 动作 max abs | **0.0013 – 0.0037 rad** |
| mean abs | 0.00032 – 0.00065 |
| 夹爪 max abs | 0.0006 – 0.0015 |

残差来源：参数以 bf16 存储、离线与在线各经过一次 jit 的重排、float64→float32 反归一化顺序差异；
量级 <5e-3 rad，符合预先设定的容差（见 `contract_test_plan.md`）。**PASS**。

同时确认 `Policy.metadata` 中记录的 checkpoint 路径就是 `/scratch/lijh/exp47/merged/step3000`
（不是 base 或其他 checkpoint）。

## 5.2 五类误差分开报告（不合并成单一 MAE）

| # | 误差类别 | 数值 | 来源/证据 |
| --- | --- | --- | --- |
| 1 | 训练/推理接口差异 | **0.0**（324/324 字段逐位一致）；推理实现差异 <3.7e-3 rad | Gate 1、Gate 5 |
| 2 | 模型预测 vs 专家标签 | chunk k0 MAE **0.019 rad**、k1–15 **0.024 rad**、16 步末端最大 **0.37 rad** | 4.6 `offline/step3000_stats.json` / `_chunk.json` |
| 3 | Adapter / 限速对目标的影响 | 专家轨迹上 **0.0 rad，0/707 次触发限速**；k8–15 有 0.35% 行被裁剪（max 0.013 rad） | Gate 2/3 |
| 4 | 控制器跟踪误差 | max **0.0121 rad**、mean 0.0017 rad（707 tick） | Gate 3 trace |
| 5 | 闭环状态漂移 | 第 **1–2** tick 起 EE 偏离专家 >3 cm 且持续；45 tick 内最大 0.33–0.68 m | Gate 4 `first_deviation_v2.csv` |

## 5.3 判读

- 接口层（1）与执行层（3、4）都在毫米/毫弧度级，**不构成闭环失败原因**；
- 主导项是（2）模型预测与专家标签的差异，以及它引发的（5）状态漂移：
  离线 chunk 0.019 rad 的误差，在 20 Hz 闭环里被**积分放大**到 3 cm 级/1–2 tick；
- 因此"契约通过"并不等于"模型能完成技能"：剩余差距属于**闭环学习/终态精度**问题，
  而不是训练—推理—执行链路的数值不一致。
