# Gate 0：修复后执行契约核查（基准 `e128b35` + 本轮两处补丁）

## 1. 版本与关键参数

| 项 | 值 |
| --- | --- |
| Simulate HEAD（核查时） | `e128b35`（本轮修复另提交，见 `artifact_manifest.json`） |
| OpenPI | `215abfb217dbac7d5f1273282331b9b1866c0479` |
| 数据集 | 4.1 的 90 条（train 61 / val 16 / test 13），`exp4.1-split-v1` |
| checkpoint | step501（502 次更新）/ step901（902 次更新）：LoRA npz md5 `864419…` / `7cd19b…` |
| 训练动作契约 | `exp4.1-action-v1` + `exp4.1-training-export-v1`（20 Hz，12 物理步/控制周期） |
| 评价器 | `exp4.1-strict-v1.2` |
| `execute_k` | 8（另做 1 的单变量诊断） |
| `action_gain` | 1.0 |
| `anchor_reference` | **fixed**（4.2 全部正式入口） |
| `gripper_mode` | **continuous**（0.425 → 0.017 m） |

## 2. 逐项核查（Gate 0 的 7 个问题）

| # | 问题 | 结论 | 证据 |
| --- | --- | --- | --- |
| 1 | 固定锚点是否在每个 chunk 开始时建立 | 是 | `eval_rollouts.py` 在每个 chunk 起点取**实测 q** 作锚点，`gate1_expert_replay.py` 同 |
| 2 | chunk 内是否错误更新锚点 | 否 | `chunk_to_targets(fixed)` 内 `q` 保持 `q0`，只有 recursive 才更新（`test_contract_p0c.py` 第 1、2 项） |
| 3 | 新 chunk 是否用真实观测重锚 | 是 | 单测第 3 项：新锚点 `q1=0.5` → `target=q1+a_0` |
| 4 | 控制器收到的目标是否等于预期 | 是 | Gate 1 的 teacher-forced 误差 **5.6e-17 rad**（measured）/ 2.5e-5（recorded） |
| 5 | `gripper_target_m` 单位/范围/左右指 | 米；0.017（闭合）–0.04（张开）；两指同目标 | 单测第 5–7 项；`adapt_action` 用 `g*0.04`，`binary` 时 0/0.04 |
| 6 | 20 Hz 与 240 Hz 对应 | 12 物理步/控制周期；1 tick = 50 ms | `STEPS_PER_TICK = 240//20`，`eval_rollouts` 与回放一致 |
| 7 | 训练/部署/物理是否同约定 | **本轮发现并修复最后一个不一致（见 §3）** | Gate 1 全部回放通过 |

## 3. 本轮新发现并修复的契约缺陷 #3：安全裁剪与"累计偏移"不匹配

**现象**：Gate 1 第一次回放（只有 fixed-anchor + 米单位）时，adapter 目标与专家目标差 **0.30–0.54 rad**，
4 条回放里 2 条严格失败（A90 ep5 `tilt_ok`、ep6 `yaw_ok`；B90 ep0 `wall_contact_ever`）。

**根因**：`max_delta_rad = 0.25` 是按"**每步增量**"设计的，但 fixed-anchor 下 `a_k` 是"**跨 k+1 步的累计偏移**"。
A90 旋转阶段专家每步转 ~0.04 rad，第 8 步的 0.31 rad 合法偏移被 0.25 截断 → 目标系统性落后。

**修复**：`chunk_to_targets(anchor_reference='fixed')` 对第 k 步使用 `max_delta_rad * (k+1)` 的上限
（`recursive` 保持原样，4.0 归档结果不受影响；`test_contract_p0c.py` 第 4、5 项覆盖）。

**修复后**：teacher-forced 误差降到 **5.6e-17 rad**，8 条回放全部严格成功（见 `expert_replay_results.csv`）。

## 4. 正式入口的一致性检查

| 入口 | anchor | gripper | 备注 |
| --- | --- | --- | --- |
| `scripts/exp42/eval_rollouts.py`（4.2 闭环） | fixed | continuous | 本轮重跑 |
| `scripts/exp42/gate1_expert_replay.py`（专家回放） | fixed | continuous | 本轮新增 |
| `scripts/exp40/level4_closed_loop.py`（4.0 归档闭环） | 逐步重锚（等价 fixed，但每步用实测 q） | binary | 历史结果保留，未改 |
| `compare_actions.py` | 不执行动作 | — | 只比较 chunk |

> 4.0 的 `level4` 用"每步实测 q 作锚点"（等价于每步重锚），与 4.2 的"chunk 起点锚点"不同；
> 两者都在各自的数据契约下自洽，但**不能混用结果**。已在报告中标注。

## 5. 单元测试

`scripts/exp42/test_contract_p0c.py`（纯 CPU，9 项全过）：

1. fixed anchor `target[k]=q0+a_k`；
2. recursive 逐级累加（旧行为可复现）；
3. 新 chunk 用实测 q 重锚；
4. fixed 下第 k 步允许 `(k+1)*cap`（0.3 rad 不再被截断）；
5. recursive 仍按 0.25 截断；
6. 夹爪 continuous 0.425 → 0.017 m；
7. 夹爪归一化值 0.425 不能被当作米；
8. binary 0.425 → 0.0 m（闭合）；
9. 固定锚点精确恢复专家目标（误差 0）。

## 6. P0 安全与可观测性补强（2026-11-04，回应独立审计）

| 项 | 状态 | 证据 |
| --- | --- | --- |
| Adapter `accepted=False` 时硬阻断 | **已实现** | `eval_rollouts.py` 执行前检查；触发则停止本回合并记录 `stopped_episode` |
| 逐控制周期限速（相邻命令跳变） | **已实现** | `--max-joint-step-rad` 默认 0.30；`limit_joint_step()` 单测 3 项 |
| 逐子步日志（不只 chunk 首行） | **已实现** | `*_epXX_substeps.csv`（25 列，含 q_target/q_after/dq/EE/物体/接触/accepted/clipped） |
| 完整 chunk 记录 | **已实现** | 每条 tick 存 `chunk_rows`（8×8） |
| 单测 | **12/12 通过** | `contract_unit_tests.json` |

重跑（`rollouts_p0_logged/`）：**拒绝 0 次、限速 0 次**（最大跳变 0.25 rad），
说明此前失败不是拒绝/限速造成的；逐子步配对结论见
[`p0_substep_analysis.md`](p0_substep_analysis.md)。

