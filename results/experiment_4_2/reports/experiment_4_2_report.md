# 实验 4.2 报告：π₀.₅ 原子技能联合 LoRA 微调与自主物理评价

- Simulate 基准：`b1b1c41`（4.1 最终）；4.2 代码/工件见 `artifact_manifest.json`
- OpenPI：`215abfb217dbac7d5f1273282331b9b1866c0479`，checkpoint `pi05_droid`（11.576 GiB, quantile）
- 相机：`exp4.0-camera-v3` / SHA256 `418bad2f33ef5a9e`
- 动作契约：`exp4.1-action-v1` + `exp4.1-training-export-v1`（20 Hz）
- 评价器：`exp4.1-strict-v1.2`
- 硬件：1× A40 48 GB（GPU0 训练、GPU1 推理），80 核 / 251 GB RAM

## 0. 结论

**训练链路完全打通并真实完成；原子技能尚未学会。**

| Gate | 内容 | 判定 |
| --- | --- | --- |
| 1 | OpenPI training batch（真实数据 + 真实 loss） | **PASS**（附环境变通） |
| 2 | Action / time / normalization 契约 | **PASS** |
| 3 | LoRA 训练（梯度更新、可恢复、checkpoint） | **PASS** |
| 4 | Checkpoint 推理（加载微调权重并控制 Panda） | **PASS** |
| 5 | 自主原子评价 | **PARTIAL**（机制通过，技能未达成：0/8；P0 诊断显示幅度已收敛到 1.5–2×，但方向余弦 ≈ 0） |
| 6 | 可复现性 | **PASS**（配置/日志/计划/manifest 已提交） |

| 原子技能 | 判定 | 依据 |
| --- | --- | --- |
| `A90_ATOMIC_SKILL_LEARNED` | **NO** | 4 条 A90 rollout：0 成功；最好的一条接触 69 个物理步、抬升 8.2 mm（阈值 50 mm），方块被推走而不是被抓起 |
| `B90_ATOMIC_SKILL_LEARNED` | **NO** | 4 条 B90 rollout：0 成功；2 条完全没接触，1 条接触 22 步但把方块推了 **−28.5 mm**（方向反了） |

## 1. Gate 1：真实训练 batch（PASS，附环境变通）

**环境阻碍（已记录）**：CentOS 7 / glibc 2.17 上装不了 `lerobot`，因此
`openpi.training.data_loader` **无法导入**，官方 `scripts/train.py` 入口不可用
（`environment_manifest.json: environment_blocker`）。
4.0 的推理靠一个 `lerobot_stub` 绕过；训练侧没有任何 stub 能提供真实数据集，因此本轮：

- **复用官方全部训练组件**：`pi05_droid_finetune` 的模型配置（`pi05=True, action_dim=32,
  action_horizon=16`）、`droid_policy.DroidInputs/DroidOutputs`、`Normalize(quantile, DROID stats)`、
  `TokenizePrompt`、`PadStatesAndActions(32)`、`pi0.compute_loss`、官方 optimizer/lr schedule；
- **只替换数据迭代器**：把 4.1 的 90 条 episode 导出成 npy/npz（5.9 GB），按 `exp4.1-split-v1`
  窗口采样（horizon 16，stride 4，train 61 条 → **2521 个窗口**）。

真实 batch（batch=4）实测：

| 项 | 值 |
| --- | --- |
| state | `(4, 32) float64`（8 维物理 + 24 padding） |
| actions | `(4, 16, 32) float64`（horizon 16，前 8 维有效） |
| images | `base_0_rgb (4,224,224,3) float32`、`left_wrist_0_rgb (4,224,224,3)`、`right_wrist_0_rgb` mask=**False** |
| tokenized_prompt | `(4, 200)`，mask 求和 285 |
| action 有效步 | 16/16 |
| batch 内技能 | 同时含 A90 与 B90 |
| forward loss | **1.1643**（有限，非 NaN） |

## 2. Gate 2：动作/时间/归一化契约（PASS）

完整链路与逐条回答见 [`training_data_contract.md`](training_data_contract.md)。要点：

- 训练标签 = **每控制步关节增量（rad/step, 20 Hz）** + 连续归一化夹爪；**没有二次 delta**，
  **没有 0.2 系数**，训练与推理同单位同频率；
- 夹爪：state 用观测（`mean(fingers)/0.04`），action 用**专家下发的目标**（0.425 闭合 / 1.0 张开）；
  部署端 0.5 二值化与之一致（45/45 A90 episode 都存在"观测张开、动作已闭合"的帧）；
- 8D→32D：`PadStatesAndActions` 零填充，**没有 mask**，padding 维以 0 为目标参与 loss
  ——这就是官方 `pi05_droid` 的训练方式；输出侧 `DroidOutputs` 只取前 8 维；
- 归一化：**用官方 DROID stats**。train-split 前 7 维归一化后 clip 比例 **0.000**；
  夹爪维 0.28–0.31 的"clip"是 1.0 略超 q99(0.9998) 的阈值效应（差 0.0004），不是尺度错误
  （[`normalization_decision.md`](normalization_decision.md)）。

## 3. Gate 3：LoRA 训练（PASS）

| 项 | 值 |
| --- | --- |
| 可训练参数 | **466.96 M / 3403.4 M = 13.72%**（LoRA 变体 `gemma_2b_lora` + `gemma_300m_lora`，EMA 关闭） |
| 训练步数 | **501 + 400 = 901**（两段：初训 + 续训，续训从 LoRA npz 恢复，51/51 张量成功注入） |
| batch / 有效 batch | 4 / 4 |
| 吞吐 | **0.44–0.55 step/s**（2.25–2.62 s/step，含数据加载）；compile 15–16 s |
| 等效 epoch | 0.79 + 0.64 = **1.43**（2521 窗口 × horizon 16） |
| loss | 1.16 → **0.027**（train）/ **0.0265**（val，step 400 续训） |
| 梯度 | 非零（grad_norm 0.4–88，无 NaN） |
| checkpoint | LoRA npz @100/200/300/400/500/501（每份 1.73 GB）+ 续训 200/400/401；完整 params（orbax，12.6 GB）2 份 |
| 可恢复 | 已用 `--init-lora` 验证：从 step501 继续训练 400 步 |

`training_speed_benchmark.json` 记录了 step 耗时、compile 时间、吞吐、GPU 状态；
`training_logs/train_step501.log`、`train_step901.log` 保存逐步 loss/grad/eval。

## 4. Gate 4：Checkpoint 推理（PASS）

- LoRA 参数合并成完整 params（51/51 张量，`export_params_from_lora.py`），嵌套为官方
  `{"params": tree}` 格式，并链接 `assets/droid`；
- 用与训练**完全相同**的注册配置 `exp42_pi05_droid_lora`（否则参数树不匹配）通过官方
  `create_trained_policy` 加载，起 4.0 的 websocket 服务，20 Hz 闭环跑满 45 tick × 8 执行步；
- 同观测对比（[`action_compare_baseline_vs_finetuned.json`](action_compare_baseline_vs_finetuned.json)）：

| 样本 | baseline absmax | finetuned absmax | 差异 absmax | 相同？ |
| --- | --- | --- | --- | --- |
| A90 帧 | 0.9969 | 0.9475 | **0.5642** | 否 |
| B90 帧 | 0.3640 | 0.5718 | **0.4272** | 否 |

→ 微调权重确实生效（不是误加载原始 checkpoint）。

## 5. Gate 5：自主原子评价（PARTIAL：0/8）

两批 checkpoint，各 2 A90 + 2 B90，45 tick（20 Hz，execute_k=8），seed 9000/9001，
同一套场景/相机/评价器（`exp4.1-strict-v1.2`）：

| checkpoint | A90 成功 | B90 成功 | 机器人-方块接触步 | 现象 |
| --- | --- | --- | --- | --- |
| step 501 | 0/2 | 0/2 | **0, 0, 0, 0** | 机械臂大幅摆动但从未接触方块 |
| step 901 | 0/2 | 0/2 | **69, 21, 0, 22** | 开始接触：A90 把方块推走 8 cm、抬升 8.2 mm；B90 一条把方块推了 **−28.5 mm**（方向反） |

失败原因分布（8 条）：`grasped_by_two_fingers` 3、`pushed_by_robot` 4、`advance_ok` 1、
`lifted` 1（部分重叠：一条只记首个不满足项）。

**明确记录**：

1. **基线闭环没有跑**（时间预算用尽），因此不能给出"微调 vs 原始 `pi05_droid`"的配对成功率差；
2. n=2/技能 → 任何成功率都不具统计意义（Wilson 95% 区间下界 0、上界约 0.66）；
3. 评测条件在跑之前固定（场景、seed、tick 数、execute_k、评价器版本），没有为微调模型调阈值。

## 6. 诊断：loss 降了，为什么物理成功率是 0

> **本节结论已由 P0 诊断修正**（`diagnostics_p0.md`，review 后补充）。
> 原来的"幅度过大是唯一根因"**不成立**：幅度只解释了一半，另一半是**方向**。

### 6.1 专家观测上的动作对比（10 帧：6 train + 4 val）

| 模型 | 幅度比（模型/专家） | 关节 MAE | 首步方向余弦 | 0.25 rad 裁剪比例 | 夹爪 MAE |
| --- | --- | --- | --- | --- | --- |
| 原始 `pi05_droid` | **4.98×** | 0.154 | 0.11 | **13.3%** | 0.406 |
| step501 | **1.54×** | 0.066 | 0.04 | 0.18% | **0.018** |
| step901 | **2.02×** | 0.061 | **−0.0001** | 1.5% | 0.034 |

→ 微调**确实在学习**：幅度从 5× 压到 1.5–2×，关节 MAE 降 60%，夹爪几乎学会；
**但逐帧 7 维方向余弦 ≈ 0**（个别为负）——学到了"步幅多大"，没学到"往哪走"。

### 6.2 loss 归因：低 loss 与真实动作维度几乎无关

把真实 8 维动作目标置零后重算 loss（24 维 padding 的目标本来就是 0）：

| 模型 | 真实目标 loss | 真实维度置零 loss | 差值 |
| --- | --- | --- | --- |
| step501 | 0.05727 | 0.05654 | **+0.0007** |
| step901 | 0.04032 | 0.04438 | **−0.0041** |

`pi0.compute_loss` 对全部 32 维求均值且**没有 action mask**（官方行为），
而本次对照显示 loss 对真实 8 维几乎不敏感 → **0.027 的验证 loss 不能作为"学会了"的证据**。
这与 review 的判断一致，也修正了 §5 里"loss 降了但行为不对"的单一解释。

### 6.3 裁剪与方向

- 裁剪不是主因：微调模型只有 0.18%/1.5% 的步被裁；**原始模型才是 13.3%**
  （此前"无异常裁剪"的表述只对微调模型成立，已更正）；
- B90 出现过 −28.5 mm 反向位移，与"方向余弦 ≈ 0"一致，不是单纯幅度问题。

### 6.4 P0b 复核（运动帧 / loss 归因 / 初始状态）

详见 [`diagnostics_p0b.md`](diagnostics_p0b.md)。三个结论：

1. **方向余弦≈0 是抽样伪影**：只在专家运动帧（|Δq|≥0.02）上算，A90 余弦 0.06→0.37→**0.46**，
   B90 −0.05→0.34→0.23；幅度比 0.97–1.21×，MAE 0.037–0.073 rad；
2. **"padding 是主因"未证实**：24D padding 每维 loss 0.176（关节 0.227），其 flow-matching 目标
   是噪声而不是 0；它占平均 loss 的 ~69% 主要靠维数占 75%，而关节梯度范数 6.56 与 total 6.72 相当；
3. **闭环真正的第一道坎是初始 HOME 帧**：专家起始帧 Δq≈8.8e-6（静止），
   而 baseline/step501/step901 都输出 0.27–0.54 rad；增益 0.5/0.2 也无效（物体完全不移动）；
   原始 `pi05_droid` 基线同样 0/4 且物体未动。数据里 31–49% 的帧近静止，
   但模型在初始观测下不输出静止动作。

### 6.5 结论

需要修的是 **loss 的有效维度（7D/1D 分组或 action mask）与方向监督**，
而不是单纯把步数从 901 加到几千。P1（动作增益 1.0/0.2/0.1、基线闭环）本轮**未完成**（时间预算）。


> **审计后更正（2026-11-03）**：Gate 5 与 §6 的 8 条 rollout 是在**有缺陷的执行契约**上跑的
> （chunk 锚点递归累加 + 夹爪归一化值当米下发）。独立审计确认了这两个缺陷，已在
> `src/exp40/action_adapter.py`（新增 `anchor_reference='fixed'`）与
> `scripts/exp42/eval_rollouts.py`（改用 `gripper_target_m` + 连续夹爪）中修复并重跑：
> 修复后仍 0/4，但行为从"大幅乱扫"变为"持续接触（B90 616 个物理步）但方向错"。
> 详见 [`independent_audit/agent_verification.md`](../independent_audit/agent_verification.md)。
> 修复前的 rollout 仅作历史对照，不与修复后混算。

> **后续诊断（contract_replay_p0c）**：修复执行契约后，专家动作经 Adapter 的物理回放
> **8/8 严格成功、目标误差 1e-17**（Gate 1 PASS）；B90 的 +Y 被定位为**模型 EE 方向**本身
> （接触期模型 Δy=+0.072 vs 专家 −0.000），不是接触几何；启动段首步监督明确、chunk 行存在结构性歧义。
> P0c 训练因条件不满足**未开展**。详见
> [`../contract_replay_p0c/final_report.md`](../contract_replay_p0c/final_report.md)。

## 7. RQ 回答（对应 §十六）

**RQ1（A90 能在多少目标上严格成功）**：0/4 rollout（2 个目标 × 2 seed），无一成功。
**RQ2（B90 在多少条件下能真正推墙）**：0/4；2 条未接触，1 条接触但反方向。
**RQ3（真实接触/目标到达/稳定结束比例）**：接触率 3/8；到达目标区 0/8；稳定结束 8/8（都停住了）。
**RQ4（严格评价器相对旧判据改变了多少标签）**：本轮 8 条 rollout 旧"近墙 2 cm"判据同样全失败
（方块与墙的间隙 0.17 m 以上），所以差异为 0；严格判据的价值在"接触步 = 0 时直接判 `pushed_by_robot`"，
这在旧判据里无法区分。
**RQ5（动作表示是否匹配 π₀.₅）**：契约层面匹配（Gate 2 PASS）；**数值尺度层面不匹配**（见 §6）。
**RQ6（数据是否存在重复/窄分布）**：是——A90 的 45 条只覆盖 9 个目标区域、Δq 分布很窄（q99=0.05 rad），
这既限制了多样性也放大了尺度不匹配的影响。
**RQ7（是否可构造无泄漏的组合测试条件）**：可以（`split_manifest` 已按目标/条件组冻结；
组合测试将仅持有配对关系），但**前提是先把单技能学会**。
**RQ8（进入 4.3 前的风险）**：动作尺度/训练步数、无基线闭环、样本量、B90 方向错误。

## 8. Observed / Supported / Not Verified / Blockers

### Observed
- 真实 OpenPI batch（4,32）/（4,16,32）/ 双路 224² 图像 / token mask 285 / loss 1.1643；
- LoRA 466.96 M 可训练（13.72%），901 步，loss 1.16 → 0.027，val 0.0265，0.44–0.55 step/s；
- checkpoint 可保存/恢复（LoRA npz + 完整 orbax params），可被官方 `create_trained_policy` 加载；
- 闭环 8 条 rollout：0 严格成功；从"完全不接触"到"接触但方向/抓取错误"有明显进展；
- 推理 0.23–0.37 s/chunk，45 tick 全程无 NaN/无安全停止。

### Supported
1. 在**没有 lerobot** 的 CentOS 7 上，用官方模型/transform/optimizer/loss + 自写数据迭代器，
   可以完整复现 π₀.₅ 的 LoRA 训练（Gate 1/3 PASS）。
2. 微调权重确实改变策略输出（同观测动作差 0.43–0.56），并且能在 4.0 的推理闭环里真实控制 Panda。
3. 当前失败不是"训练没跑通"，而是**动作幅度尺度**与训练分布不匹配（0.2–0.5 vs 0.05 rad/step）。

### Not Verified
1. 原始 `pi05_droid` 的闭环成功率（基线未跑）；
2. 训练到 2–6 epoch 后是否收敛到正确尺度（本轮 1.43 epoch）；
3. 训练侧动作重标定（缩放 Δq）是否能解决尺度问题；
4. 更大的 batch / 更长 horizon / 更多数据的影响。

### Blockers
| # | 阻塞项 | 影响 | 建议 |
| --- | --- | --- | --- |
| B1 | 动作尺度不匹配（模型输出 4–10× 训练分布） | 无法接近/抓取/定向推 | 继续训练到 2–3k step，并在闭环前加"输出分位数门禁" |
| B2 | 没有基线闭环 | 无法归因"微调是否带来增益" | 用同 seed 跑 baseline 4 条 |
| B3 | 样本量小（n=2/技能） | 成功率不可解释 | 扩到 15–20 条/技能（4.2 续） |
| B4 | lerobot 缺失 | 官方 `train.py` 入口不可用 | 已用自写迭代器绕过；若要严格复现官方入口需换容器 |
| B5 | 磁盘只剩 ~230 GB | 多个 12.6 GB checkpoint 会吃紧 | 只保留最终 + 最佳 checkpoint |

## 9. 复现

```bash
# 1) 导出训练数组（simulate venv）
python scripts/exp42/export_data.py --data data/exp41 --out /scratch/lijh/exp42/data
# 2) 训练（openpi venv，GPU0）
bash server/exp42/run_train.sh train --steps 501 --batch 4 --save-every 100 --export-params
# 3) 续训（可选）
... --init-lora <lora_step00501.npz> --steps 401
# 4) 合并 LoRA → 完整 params（嵌套 {"params": ...}）
python scripts/exp42/export_params_from_lora.py --lora <npz> --out <dir>/params
# 5) 动作对比（Gate 4）
python scripts/exp42/compare_actions.py --ckpt <dir>
# 6) 闭环 + 严格评价（Gate 5）
bash server/exp42/run_eval.sh <dir> 8124 2 45 1
```
