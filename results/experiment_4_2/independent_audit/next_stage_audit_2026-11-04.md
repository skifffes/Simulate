# Simulate π₀.₅ — 实验 4.2 独立源码审计（阶段报告）
日期：2026-11-04  
固定审计版本：`4a69c95`（通过 GitHub Raw 阅读该固定版本文件）；**未通过 git 命令核实当前 main HEAD / 工作树状态**。  
审计方式：远端公开源码与仓库归档的 CSV/JSON 只读交叉核对；本环境未连接服务器、未执行 π₀.₅ 或 PyBullet、未启动训练。  
科研边界：A90/B90 原子技能先达标，之后才能研究不含 A→B 组合示范的零样本技能组合与状态约束。

## 1. 已读取的源码与数据

- [`scripts/exp42/train_lora.py`](https://raw.githubusercontent.com/skifffes/Simulate/4a69c95/scripts/exp42/train_lora.py)：`windows`、`make_item`、`make_transforms`、`make_train_step`、`--init-lora`、更新计数。
- [`scripts/exp42/eval_rollouts.py`](https://raw.githubusercontent.com/skifffes/Simulate/4a69c95/scripts/exp42/eval_rollouts.py)：动作推理、固定锚点、夹爪控制、执行日志与严格评价。
- [`src/exp40/action_adapter.py`](https://raw.githubusercontent.com/skifffes/Simulate/4a69c95/src/exp40/action_adapter.py)：`adapt_action`、`chunk_to_targets`、拒绝/裁剪语义。
- [`src/exp41/training_export.py`](https://raw.githubusercontent.com/skifffes/Simulate/4a69c95/src/exp41/training_export.py)：逐控制步数据契约。
- [`src/exp41/run41.py`](https://raw.githubusercontent.com/skifffes/Simulate/4a69c95/src/exp41/run41.py)：采集 prompt、20Hz/240Hz、A90 目标抖动与初始录帧。
- [`scripts/exp42/export_data.py`](https://raw.githubusercontent.com/skifffes/Simulate/4a69c95/scripts/exp42/export_data.py)：真实存储的 state/action、两路 256×256 视频、prompt 与 split。
- [`results/experiment_4_2/contract_replay_p0c/expert_replay_results.csv`](https://raw.githubusercontent.com/skifffes/Simulate/4a69c95/results/experiment_4_2/contract_replay_p0c/expert_replay_results.csv)：专家/Adapter 物理回放原始记录。
- [`results/experiment_4_2/rollouts_fixed_contract/B90_ep00.json`](https://raw.githubusercontent.com/skifffes/Simulate/4a69c95/results/experiment_4_2/rollouts_fixed_contract/B90_ep00.json)：自主模型失败的原始指标。
- [`scripts/exp42/b90_direction_analysis.py`](https://raw.githubusercontent.com/skifffes/Simulate/4a69c95/scripts/exp42/b90_direction_analysis.py) 及对应 [方向报告](https://raw.githubusercontent.com/skifffes/Simulate/4a69c95/results/experiment_4_2/contract_replay_p0c/b90_direction_analysis.md)。
- [`results/experiment_4_1/dataset/split_manifest.json`](https://raw.githubusercontent.com/skifffes/Simulate/4a69c95/results/experiment_4_1/dataset/split_manifest.json)。

## 2. 验收表

| 项目 | 结论 | 代码或原始证据 / 界限 |
| --- | --- | --- |
| 实际 HEAD 和本地工作树 | NOT VERIFIED | Git 远程操作 DNS 失败；只能核对 `4a69c95` 指定文件内容 |
| Fixed-anchor 进入正式闭环 | PASS（静态源码） | `eval_rollouts.main()` 传 `anchor_reference='fixed'`，`chunk_to_targets` 每行相对于 `q0` |
| 每个 chunk 的重新锚定 | PASS（静态源码） | 每次推理前读取当前测量 `scene.sim.q()` |
| Chunk 边界连续性 | PARTIAL | 无完整相邻步 `q_target`/`q_measured` 日志，不能验证跳变 |
| Gripper 物理米单位 | PASS（静态源码/归档回放） | `continuous` + `adapted[i].gripper_target_m`，连续映射 `g*0.04` |
| 累计偏移裁剪 | PASS（语义修复） | 固定锚点 k 行按 `0.25*(k+1)` 限制相对 `q0` 的累计偏移 |
| 单步动态安全边界 | PARTIAL（存在缺口） | 没有检查相邻命令差、速度、加速度；见第 4 节 |
| Adapter 拒绝动作的强制阻断 | BLOCKED | `adapt_action.accepted=False` 时 `eval_rollouts` 未检查，仍执行裁剪后的目标 |
| 专家动作恢复 | PASS（有限样本/归档证据） | A90 ep5/6、B90 ep0/1；measured-anchor 目标误差 ~5.6e-17 / 2.8e-17 rad |
| 连续物理专家回放 | PASS（有限样本） | 4 个独立 episode × 2 种锚点回放方式，8/8 严格成功，**不是** 8 个独立初始状态 |
| B90 真实墙接触检测 | PASS（专家样本） | 专家 B90 ep0/1 墙面接触 669 / 922 个物理步；自主 ep00 为 0 |
| B90 模型 EE 方向归因 | PARTIAL | 实际 EE +Y 可观测，但预测 L1 只用 chunk 首行，执行 L2 用前 8 行；未逐步分离 Adapter 与控制 |
| HOME 监督分析 | PARTIAL | 90 条前 6 帧是程序等待；首行动作监督明确、后续多步未来存在可观测状态近似别名 |
| Checkpoint/资产真实哈希 | NOT VERIFIED（本环境） | 仓库 manifest 给出路径/历史 MD5；无法读服务器权重与当前规范化资产 |
| 原子自主任务能力 | BLOCKED | 既有修复后 rollout 未通过严格 A90/B90；不能进入组合技能结论 |

## 3. 已确认的三项历史执行错误已修复

1. 训练 chunk 行含义：`a_k = A[t+k,:7] - S[t,:7]`，部署应执行 `q_target[k] = q0 + a_k`，不能递归累加。
2. 模型夹爪输出 0~1 是比例，连续执行必须换算到单指位移米：`g*0.04`，不能把 0.425 直接作为 0.425m。
3. `max_delta_rad=0.25` 是逐步限制，不能逐行裁剪跨 k+1 步累计量；源码已将 fixed 模式限制扩展为 `0.25*(k+1)`。

上述修复在 `4a69c95` 的共享 Adapter 及正式自主闭环源码中都能确认。但修复表示 **训练-部署标签约定相符**，不表示模型学会了动作，也不表示控制器动态安全。

## 4. 新发现的确定问题

**P0 / 运行安全**：`adapt_action()` 的 docstring 规定 `accepted=False` 时调用方应停止并记录；`eval_rollouts.py` 的 `for i in range(k)` 无检查，直接 `scene.sim.command(adapted[i].q_target)`。`accepted=False` 可由超官方上限、超关节限位或非有限值触发。是否发生在历史 rollout 中无逐行日志可判定。

**P0 / 动态连续性**：固定锚点逐行最大偏移 `0.25*(k+1)` 不能约束 `q_target[k]-q_target[k-1]`。例如 q0=0，前两行偏移分别为 +0.25、-0.50，均符合相对固定锚点的限制，连续命令跳变却为 -0.75 rad。应另加逐控制周期的限速/限加速度约束或拒绝机制并记录。

**P0 / 诊断可辨识性**：rollout 的 `action_log` 只保存首行、最大幅度及每个 chunk 执行后 EE/物体位移；不保存完整前 8 行预测/反归一化结果、每步 clipped_dims/reasons/accepted、每步实际 q/qd、控制跟踪差。因此既有 B90 L1=`Jacobian @ chunk_first_row` 与 L2=整个 8 步真实位移不可一对一归因。

**P1 / 训练 action 契约差异**：`src/exp41/training_export.py` 的 7D 单步动作定义为 `A[t]-S[t]`；实际 `train_lora.make_item()` 对 k>0 使用 `A[t+k]-S[t]`。这是被 fixed-anchor 执行器适配的多步目标定义，但不等同 DROID 预训练的逐控制步 `joint_velocity` 动作语义。标记 SUPPORTED（语义差异已证实），而非根因已证实。

**P1 / normalization 统计口径**：`normalization_decision.md` 中 Simulate vs DROID 的关节尺度使用逐控制步 Δq。不能将 2.4%–6.1% 直接当作真正训练 16-step 固定锚点各行的相对跨度；需按 `make_item()` 展开所有训练窗口后逐 k、逐关节统计，并验证 normalized/physical 双空间方向误差。

**P2 / 参数行为**：`train_lora.py` 宣称 `--accum`，但实际 optimizer update 仅每 batch 一次；该参数只进入 benchmark 的记录，未实现梯度累积。默认 accum=1 不影响既有 baseline，若实际设为4却按有效 batch16 解释则不成立。`--init-lora` 只恢复模型 LoRA 权重，optimizer state 和学习率调度重新初始化；历史 step901 应理解为两个阶段累计更新标签，而非完全连续的同一优化过程。

## 5. 原始结果复核与误读纠正

- 固定版本自主 `B90_ep00.json`：严格失败，`advance_x_m=0.014567`、`lateral_drift_m=0.052438`、`wall_contact_steps=0`。
- 固定版本专家 `expert_replay_results.csv`：B90 ep0 专家位移 +0.18365m、墙面接触669个物理步；ep1 +0.17597m、墙面接触922步。专家动作能被 Adapter 在这些代表性条件下重放，不等于自主模型成功。
- B90 ep0 的低 y 区目标与 home 构型属于训练条件，而非 val/test 新目标；失败不需要 A→B 组合分布偏移解释。
- 回放日志 `final_q_vs_expert_state_max_err_rad` 是专家与 Adapter 最终状态一致性，不是目标跟踪误差。B90 ep1 `max_q_track_err_rad≈0.092586`，因此“控制器追踪精度 1e-8rad”是误读。
- 以前 B90 +Y 报告证明实际 EE 与物体横向运动；但只用首行动作估计 L1，无法彻底排除后续动作、裁剪、控制器跟踪和自身状态分布偏移的影响。

## 6. 阶段结论（证据标签）

- **OBSERVED**：三项历史执行错误的修补逻辑出现在固定版本的正式执行入口；Adapter 的 `accepted` 未被闭环检查，日志粒度不足。
- **SUPPORTED**：修复后专家能重放成功，但 π₀.₅ 自主在已知 B90 条件下仍失败；模型动作生成/条件行为尚不可靠。固定锚点训练目标与 DROID 预训练先验的语义存在差异。
- **HYPOTHESIS**：DROID 动作归一化压缩有效信号、16-step 多步表示、启动时序歧义、示范覆盖和联合条件动作学习相互影响。
- **NOT IDENTIFIABLE**：哪一个是主要剩余训练根因、预测与控制误差占比、模型是否在完整任务阶段能纠正 +Y 偏差。
- **NOT VERIFIED**：远端 main HEAD、当前服务器内 checkpoint 哈希/状态、真实模型复跑和任何新训练结果。

## 7. 直接实施的下一步（不需重新训练）

1. 给 `eval_rollouts.py` 加 `accepted/reasons/clipped_dims` 硬检查，遇到非有限值/超限停止本回合并记录；对正常安全裁剪也记录变更幅度。修改后跑 CPU 单测 + 专家物理回归测试。
2. 逐执行子步保存：原始网络32D输出、反归一化8D、Adapter原始/目标/裁剪、实测 q/qd、EE SE(3)、物体位姿速度、接触与阶段；不要只保存 chunk 首行。
3. 从同一实际关节起点对完整 8 步的 **预测 EE(FK(q_target[k]))**、实际 EE(FK(q_measured[k])) 和物体状态做配对，切分首次接触之前与之后。
4. 对训练 split 的真实 `make_item()` 全部16步标签做归一化统计，按 k、阶段和关节输出；脚本 `audit_action_dataset.py` 可直接在已有 `/scratch/lijh/exp42/data` 上只读运行。
5. 仅当上述审计不再发现更基础错误时，开启固定预算的单变量 normalization 对照（见下一份报告）。不得启动 A→B 组合评价。
