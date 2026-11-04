# Simulate / π₀.₅ Gate 4 审计补充：动作表示实验启动门槛

**审计对象**：`skifffes/Simulate` 固定提交 `0e7bb26`。  
**范围**：公开源码、已归档 CSV/JSON/报告；本地未连接训练服务器，未复跑 PyBullet 或 GPU。  
**当前结论**：执行安全补强已有源码证据；但 Gate 2 的“命令方向”统计混淆了逐步控制残差和命令轨迹，Gate 4 的“单步/原生 DROID”干预还未定义成可比较的训练-部署契约。暂不执行 300+300 重训。

## 一、已经确认

1. `0e7bb26` 存在，父提交为 `4a69c95`；本环境不能独立验证当前远端 `main` HEAD。
2. `scripts/exp42/eval_rollouts.py` 检查 `accepted=False`、非有限 `q_target`，调用 `limit_joint_step`；子步记录 `q_target/q_before/q_after`、接触、夹爪等。
3. 归档报告称三个 rollout 严格成功 0/3，拒绝与限速触发均为 0；这是归档结果，并非本地新运行。
4. `substep_fk_pairing.json` 中 B90 ep00 首次接触前模型命令相对实际状态 X 分量主导为正，接触标记子步 X 分量的平均趋势转负。
5. `action_contract_audit.json` 使用 train 61 episodes、2521 windows，锚点16步关节跨度为 DROID 的 0.175–0.761；此前逐步2.4–6.1%统计不能代表完整锚点标签。
6. A90/B90 自主严格能力仍未验证，不进入零样本 A→B 研究。

## 二、三个必须更正的解释与代码风险

### P0：`cmd_sum_dx` 不是命令轨迹位移

`/scripts/exp42/substep_fk_pairing.py`：

```python
cmd_dx = FK(q_target)[0] - FK(q_before)[0]
cmd_sum_dx = sum(cmd_dx)
```

实际统计了每步**目标末端相对当前实际末端的偏差**的总和，而不是相邻控制命令的路径位移。应并行报告：

- `target_error_i = FK(q_target_i) - FK(q_before_i)`（原指标，重命名）
- `command_path_delta_i = FK(q_target_i) - FK(q_target_{i-1})`（跨 chunk 一致，首步以前一真实起点作参考）
- `realized_path_delta_i = FK(q_after_i) - FK(q_before_i)`
- 同时记录每行接触、tick、k、首次接触后第一次新推理的时间，不能只汇总接触 flag。

因此原报告 `B90 contact cmd_sum_dx=-0.343m` 证明目标偏差趋向 -X，**不能直接读成命令目标轨迹后退34.3cm**。按原指标计算的 `cos_cmd_vs_act` 也不等价于“命令轨迹 vs 实际轨迹”的方向余弦，不能单靠它断言控制器跟踪因素已排除。

### P0：接触与重新推理不在同一时刻

从 `rollouts_p0_logged/B90_ep00_substeps.csv`：
- `tick=9,k=5`：无接触
- `tick=9,k=6`：第一次机器人-方块接触（该动作由 `tick=9` 开头、接触前的观测生成）
- `tick=9,k=7`：仍有接触
- `tick=10,k=0`：接触消失，在前一次接触之后第一次重新推理

`substep_fk_pairing.py` 根据 `touch` 逐行切 `pre_contact`、`contact`、`post_contact`，`contact` 可以是多个离散突发而非一个连续的“接触期”。B90 ep00 的 51 个 contact 标记步无法直接当作一段连续的条件行为响应。

**新增拆分**：

| 生成 chunk 时的状态 | 执行子步时的接触 | 能回答的问题 |
|---|---|---|
| 首次接触前推理 | 尚无接触 | 接近阶段预测 |
| 首次接触前推理 | 本 chunk 内才发生接触 | **开环计划不能对新接触作反应** |
| 首次接触后推理 | 无接触/接触已失 | 先前接触是否造成偏移 |
| 首次接触后推理 | 当前/新接触 | 模型是否对接触相关视觉状态产生错误命令 |

不要把“接触标记子步出错”直接等同于“接触观测导致模型反转”。

### P1：新 helper 中夹爪标签与训练不一致

`src/exp41/training_export.py::build_anchor_horizon_arrays` 目前为：

```python
out[i, :, 7] = g_action[t]
```

但真实 `train_lora.make_item()` 逐行使用：

```python
g_cmd = clip(A[t:t+16, 7] / 0.04, 0, 1)
```

应更正为：

```python
out[i, :, 7] = g_action[t:t+horizon]
```

并测试包含“接近→抓取关闭→释放张开”变化的16步窗口。此 bug 在辅助函数中，**不是已有 step901 训练失败的已证实原因**；但若下一轮把 helper 视为权威监督，会产生夹爪未来标签损坏。

## 三、为什么原计划“anchor16 vs single-step/DROID native”还不够严谨

令专家已下发的关节目标为 `A_t`，实测关节状态为 `S_t`：

| 候选表示 | 训练标签 | 对应执行器 | 能否精准重建专家命令序列 |
|---|---|---|---|
| anchor-16 | `A[t+k]-S[t]` | `fixed` | **能** |
| observed-residual-16 | `A[t+k]-S[t+k]` | `recursive` | **通常不能**；实测状态未必等于前一专家目标 |
| target-increment-16 | `k=0:A[t]-S[t]`; `k>0:A[t+k]-A[t+k-1]` | `recursive` | **能** |

举例：`S0=0, A0=1, S1=0.6, A1=1.5`：observed-residual 是 `(1,0.9)`；递归执行后第二目标为 `1.9` 而非专家 `1.5`。target-increment 是 `(1,0.5)`，递归第二目标恰为 `1.5`。

另一个语义风险：官方 OpenPI DROID 示例的 `RobotEnv(action_space="joint_velocity")`，控制频率15Hz。仓库把 `A_t-S_t` 当作“DROID 原生”并不严格等价；后者是 Panda position-control 的跟踪误差/一步位置变化，不能未经单位和控制器审计就称 velocity。

**推荐的最小可辨识干预**：

1. CPU preflight：使用 `exp42_gate4_preflight.py dataset` 扫真实 train，确认 `anchor_fixed_max` 和 `target_increment_recursive_max` 约为浮点舍入误差，统计 observed-residual recursive mismatch 与未来夹爪变化窗口比例。
2. Gate 1 回归：用真实专家动作将 anchor/fixed 与 target-increment/recursive 送入**相同 Action Adapter + PyBullet 物理控制器**，验证 A90/B90 代表性回放都严格成功、时间序列命令一致。不能仅验收最终状态。
3. Gate 2 修正：把接触前/接触中/接触后改成基于事件的连续区间和 chunk 生成时机双维分层；改正目标偏差 vs 命令路径。
4. 只有以上通过才做 300+300 的单变量训练：Group A=`anchor/fixed`；Group B=`target_increment/recursive`，其余 base/LoRA/数据次序、随机源、batch、optimizer、horizon16、DROID stats、state、gripper、32D loss、执行相同安全阈值保持一致。
5. 由于表示改变必然导致归一化数值分布改变，要先量化两组各维 normalized q01/q99/动作饱和率，**实验结论只能说“表示及其相应目标分布的联合效果”**，不能自动归因于 DROID 先验。
6. 若真正需要测试 observed-residual（接近 DROID 每步动作想法），先保持两组 `execute_k=1` 使训练/部署第一行语义一致，作为独立的下一阶段试验；不能把 observed-residual 的16行直接递归执行并声称与 anchor 在相同物理目标上比较。
7. 离线评测必须统一换算成**实际物理关节目标与 EE 位移**再比较，否则对不同参数化的原始7D MAE 进行直接比较无意义；按技能、接触前生成/接触后生成、k位置和 gripper 时序分层。
8. 只有离线时出现值得检查的差异，才使用同 checkpoint-selection 规则、同策略种子、同 A90/B90 目标和相同动作安全约束开展配对 PyBullet 自主 rollout。绝不把专家回放算自主成功。

### 关于 normalization

`anchor-16` 的整体 qspan 为 DROID 的17.5–76%，所以过去“整个训练标签都仅占2–6%”是错误的；但 `k=0` 与 observed-residual 的幅值仍可能明显更小。不能把无出界和无数值舍入误差解释成“动作信号肯定可学习”。表示对照前至少做逐 k 及按技能/接触窗口的 normalized-space 诊断，保持 normalization 参数不变，避免混入额外变量。

## 四、Gate 状态建议

| Gate | 推荐标记 |
|---|---|
| Gate0 静态安全实现 / 归档单测 | PASS (archived; not independently rerun) |
| Gate1 四条专家 episode × 两种锚点 | PASS (limited sample and archived) |
| Gate2 失败现象及接触时间归因 | **PARTIAL**；先修正 command-path 口径并重新分组 |
| Gate3 静止帧未来 chunk 歧义 | PASS (archived evidence) |
| Gate4 单变量动作表示训练 | **BLOCKED: preflight / expert replay prerequisite** |
| 原子技能真正掌握 | NOT PASSED |
| A→B 零样本组合 | NOT STARTED |

## 五、产物与引用

- 固定 GitHub commit: https://github.com/skifffes/Simulate/commit/0e7bb26
- 逐子步配对源码: https://raw.githubusercontent.com/skifffes/Simulate/0e7bb26/scripts/exp42/substep_fk_pairing.py
- 实际训练代码: https://raw.githubusercontent.com/skifffes/Simulate/0e7bb26/scripts/exp42/train_lora.py
- 夹爪辅助函数: https://raw.githubusercontent.com/skifffes/Simulate/0e7bb26/src/exp41/training_export.py
- 原始 B90 子步: https://raw.githubusercontent.com/skifffes/Simulate/0e7bb26/results/experiment_4_2/rollouts_p0_logged/B90_ep00_substeps.csv
- 官方 DROID 客户端: https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/examples/droid/main.py

**运行状态**：本轮没有对仓库写入、没有修改服务器训练进程、没有创建新的模型评测成绩。`exp42_gate4_preflight.py` 已在合成数据上通过自检；真实仓库数据分析仍需在既有服务器环境执行。
