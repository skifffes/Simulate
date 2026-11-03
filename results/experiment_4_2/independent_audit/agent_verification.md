# Agent 复核：独立审计的两项契约缺陷（已确认并修复）

审计基准 `5643db0`。本文件只做两件事：**独立复核**审计的结论、记录**修复与重跑**结果。
未训练新模型、未改严格评价器、未动 4.1 数据。

## 1. 缺陷确认（我自己的复现，不依赖审计的数字）

`scripts/exp42/verify_anchor_and_gripper.py`（纯 CPU，无 PyBullet）→ `agent_contract_verify.json`：

| 检查 | 结果 |
| --- | --- |
| 固定锚点还原合成专家轨迹（0.01→0.16 rad） | **误差 0.0 rad** |
| 递归累加（旧路径）还原同一轨迹 | **误差 1.2 rad**；末关节 1.36 rad vs 专家 0.16 rad |
| `gripper_target_m`（continuous） | **0.017 m**，与专家闭合宽度一致 |
| 下发 `gripper_value`（归一化 0.425） | 相当于 **0.425 m = 25×** |
| binary 模式 | 0.425 → 0.0 m（被当"闭合"，也不等于 0.017 m） |

结论：**审计的缺陷 #1（锚点）与 #2（夹爪单位）都成立**。
根因是我在 4.2 训练导出里用了"相对锚点"的标签（`a_k = A[t+k] - S[t]`），
而部署沿用了 4.0 的"逐级累加"（`chunk_to_targets` 旧实现），两者混用会从第 2 步起重复累加。

## 2. 修复（最小改动，不触碰 4.0 已归档结果）

| 位置 | 修复 |
| --- | --- |
| `src/exp40/action_adapter.py::chunk_to_targets` | 新增 `anchor_reference='fixed'|'recursive'`，**默认仍为 recursive**（保证 4.0 归档闭环可复现），4.2 显式传 `fixed` |
| `scripts/exp42/eval_rollouts.py` | 传 `anchor_reference='fixed'`；手指电机改用 **`gripper_target_m`**；新增 `--gripper-mode`，**默认 continuous**（0.425→0.017 m，与训练标签一致）；每条 rollout 记录 `anchor_reference/gripper_mode/execute_k/action_gain` |
| `scripts/exp42/train_lora.py` | 修正 off-by-one：`--steps N` 现在恰好 = N 次 optimizer update（旧版会多跑 1 次；已归档的 `lora_step00501/00401` 实际是 502/402 次更新） |

## 3. 修复后重跑闭环（同一 checkpoint、场景、seed、评价器）

`run_eval_fixed.sh`（GPU1，step901 的合并 params，2 A90 + 2 B90，45 tick，execute_k=8）：

| rollout | strict | 失败原因 | 机器人-方块接触步 | 物体终态 | 夹爪目标 |
| --- | --- | --- | --- | --- | --- |
| A90 ep0 | ✗ | `grasped_by_two_fingers` | **24** | (0.5450, −0.0061) | 0.0366 m |
| A90 ep1 | ✗ | `grasped_by_two_fingers` | 0 | (0.5520, 0.0051) | 0.0369 m |
| B90 ep0 | ✗ | `advance_ok` | **616** | (0.5865, **0.2903**) | 0.0336 m |
| B90 ep1 | ✗ | `advance_ok` | 21 | (0.5526, 0.2114) | 0.0207 m |

对比修复前（同 checkpoint，`rollouts_step901/`）：

| 指标 | 修复前 | 修复后 |
| --- | --- | --- |
| 夹爪实际目标 | 0.020–0.040（`gripper_value` 直接下发 + binary） | **0.021–0.037 m（连续，米）** |
| B90 最大物体位移 | **−28.5 mm（反向）** | +616 接触步、**+0.14 m 但方向偏到 +Y** |
| A90 接触步 | 69 / 21 | 24 / 0 |
| 严格成功 | 0/4 | **0/4** |

**判定**：修复消除了"累积放大"和"夹爪 25 倍"两个执行错误，行为从"大幅乱扫"变成
"持续接触但方向错"；但**没有**让原子技能成功——这与审计自己的结论一致：
HOME 首步预测偏差是独立问题（`p0c_initial_state.json`：专家静止 Δq≈1e-5，模型输出 0.27–0.54 rad）。

## 4. 受影响的已归档结果（明确标注，不隐瞒）

以下 rollout 是在**修复前**的执行路径上跑的，只能作为历史对照，不能与修复后混算：
`rollouts_step501/`、`rollouts_step901/`、`gain_rollouts/`、`baseline_rollouts/`。
修复后的结果单独放在 `rollouts_fixed_contract/`。

## 5. 审计里我未独立核对的部分

1. `scripts/exp42/independent_audit.py` 的五项契约检查我只读了脚本与输出 JSON，**未逐条复算**；
2. 审计的 `physical_contract_trace901.csv`（1344 周期、位移 0.0743→0.00855 m）我未重跑，
   只用我自己的合成测试与 4 条真实 rollout 佐证了同一机制；
3. 启动 6 帧静止是采集端 `_settle(0.3)` 产生的等待——审计这一条与我的实现一致（我确认 `pick()` 里确有 `_settle(0.3)`）。

## 6. 下一步（按优先级，仍一次只改一件事）

1. **HOME 首步**：这是修复契约后仍然失败的原因。下一步只做数据侧（首帧/静止帧过采样或首帧加权），
   32D loss、DROID stats、LR、动作增益全部不动；
2. 若首步正常，再重跑闭环（同 seed、配对 A90/B90）；
3. 之后再考虑归一化尺度（train/DROID 跨度比 0.024–0.061）与 masked-loss 在同设置下的比较。

## 7. 后续（同一轮 Gate 0/1 追加）：第三个契约缺陷

修好上面两个缺陷后，专家回放仍有 2/4 失败（teacher-forced 目标差 0.30–0.54 rad）。
根因是 `max_delta_rad = 0.25` 的**逐步**上限与固定锚点的**累计**偏移不匹配：
A90 旋转阶段第 8 步的 0.31 rad 合法偏移被截断。
已在 `chunk_to_targets(anchor_reference='fixed')` 中改为第 k 步上限 `0.25*(k+1)`（recursive 不变）。
修复后专家回放 **8/8 严格成功、目标误差 1e-17**，详见
[`../contract_replay_p0c/execution_contract_verification.md`](../contract_replay_p0c/execution_contract_verification.md)。
