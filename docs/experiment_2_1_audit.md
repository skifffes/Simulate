# 实验 2.1 · 阶段 1：核对实验 2 与冻结模型

## 1. 结论

1. 实验 2 的**模型、控制频率、相机、任务文本和成功判据都已冻结**（见 `results/experiment_2_1/frozen_config.json`）。
2. 实验 2 的 **480 条原子示范全部从同一个机器人起始构型（HOME）出发**：四个技能的起始 7 关节角各自只有 **1 个唯一值**，标准差为 0。B/C 的数据只覆盖了物体位姿变化，**没有覆盖机器人构型变化**。
3. 实验 2 的 A0/A90/C 成功集**全部有真实抓取与抬升**（最小抬升 11.4 cm）且全部自主释放；B 成功集抬升 ≈0（推动），符合预期。
4. 技能切换时**动作块确实被清空**（`SmolVLAPolicy.reset()` 重建动作队列），且切换发生在设置新任务文本之后。
5. **没有专家辅助**：策略路径不 import `SkillExpert`，唯一的 `set_gripper` 来自策略输出的动作。
6. 发现 3 个需要记录的问题（均不改变实验 2 已发布的数字）：
   - 组合评测的 A 阶段**没有记录/校验抬升**，因此无法从存档证明 80 次组合里的 A 都是"抓起来放的"而不是"推过去的"；
   - 成功判定只做一次 0.6 s 静置后的速度检查，建议 2.1 改为更长的稳定窗口；
   - 组合评测只记录终态，没有阶段事件（接近/接触/闭爪/抬升/释放），这也是 2.1 要补的。

## 2. 冻结产物

| 项 | 值 |
|---|---|
| checkpoint | `outputs/smolvla_multi/checkpoints/020000/pretrained_model` |
| model.safetensors sha256 | `b7924be757424d4ddcafb50bccb0bb28b1332a8a4f07a3cad97a93f98bfc641d` |
| 大小 / 时间 | 906 713 352 B / 2026-10-23 14:00 |
| 本地 Git commit | `cc1f9e9` |
| 动作频率 | 20 Hz（每动作保持 12 个 240 Hz 物理步） |
| 动作块 | chunk 50，`n_action_steps=50` |
| 相机 | 256×256，`supersample=1`，top + wrist |

完整配置见 `results/experiment_2_1/frozen_config.json`；本轮不更新权重。

## 3. 九项核对

| # | 检查项 | 结果 | 证据 |
|---|---|---|---|
| 1 | B/C 训练时的机器人初始状态分布 | **单一构型**：120 条 B、120 条 C 的起始关节角各只有 1 个唯一值，均等于 HOME（最大偏差 1.4×10⁻⁵，来自初始物理沉降） | `audit/summary.json` → `start_joint_stats`，`audit/start_joints.csv` |
| 2 | B/C 数据中的物体位置与 yaw 分布 | 位置：B x∈[0.538, 0.562]、y∈[0.168, 0.191]；C x∈[0.538, 0.562]、y∈[0.168, 0.192]。yaw：两组都只取 0° 与 ±90°（±6° 抖动） | 同上 `object_distributions` |
| 3 | A0/A90 实际终态关节与物体状态 | **实验 2 没有保存**（组合评测只记录物体终态，未记录关节角） | 见第 5 节；阶段 2 补齐 |
| 4 | 每个 VLA 的语言输入 | 四条固定文本，按阶段切换；组合评测在 A 成功后把 `task` 换成 B/C 文本 | `config2.TASKS`、`eval_skill_composition.py:163` |
| 5 | 切换时是否清空未执行的 action chunk | **是**。`SmolVLAPolicy.reset()` 重建 `_queues[ACTION]`；切换时先换 task 再 `runner.reset()` | lerobot `modeling_smolvla.py:354-358`；`eval_skill_composition.py:163-164` |
| 6 | 模型动作队列与推理状态 | 策略只用 `_queues` 保存动作块，`select_action` 只在队列空时重新推理；无其它跨 episode 状态 | `modeling_smolvla.py:426-448` |
| 7 | 是否有专家脚本辅助 | **没有**。策略路径不含 `SkillExpert`；`set_gripper` 只执行策略输出的夹爪目标 | `eval_multi_atomic_vla.py`、`eval_skill_composition.py` |
| 8 | 成功判定是否在物体稳定后再确认 | 有 0.6 s 纯物理静置，然后检查 `speed < 0.01`；只检查一次 | `eval_multi_atomic_vla.py:190-200`、`eval_skill_composition.py:175-180` |
| 9 | A/C 成功是否真的经历抓取、抬升、自主释放 | **原子评测：是**。A0/A90/C 的成功集抬升最小 0.114 m，全部 `released=True`；B 成功集抬升≈0（推动，合理） | `audit/summary.json` → `eval_lift` |

## 4. 对归因实验最重要的发现

### 4.1 机器人构型覆盖是一个点

B/C 的 120 条示范，起始关节角标准差全为 0。也就是说：

- 物体位姿覆盖了 4 cm × 2.4 cm（±12 mm）与 0°/90°（±6°）；
- 机器人构型**没有任何变化**，永远从 HOME 开始。

A0/A90 的终态关节构型必然不在这个集合里（A 结束后手臂在物体上方、腕部旋转），因此 T1（只恢复机器人）会是本轮信息量最大的干预。

### 4.2 组合评测缺少 A 阶段的抬升记录

实验 2 的组合脚本只记录最终物体状态，没有记录 A 阶段是否真的把物体抬起来。虽然原子评测里 A0/A90 全部是 11–13 cm 的真实抬升，但组合里的 80 次 A 成功无法从存档直接证明。2.1 的新脚本会在 A 阶段逐 tick 记录 `max_z`、抓取/释放事件，并把"抬升 ≥3 cm 且自主释放"作为 A 成功的必要条件。

### 4.3 稳定窗口偏短

0.6 s 静置 + 单次速度检查可能漏掉缓慢下沉/滑移。2.1 改为 1.0 s 静置并连续 5 个采样都满足速度阈值。

## 5. 需要阶段 2 补齐的内容

| 缺口 | 阶段 2 的处理 |
|---|---|
| A 终态关节角、夹爪、电机目标 | 用冻结模型重跑 A0/A90，保存完整状态快照 |
| A 阶段抬升与释放事件 | 逐 tick 记录 `max_z`、接触、夹爪开度 |
| 快照可恢复性 | `saveState/restoreState` + 恢复后对比关节、物体、图像 |
| 视觉因素与关节因素的分离 | T0–T3 对照，并明确说明腕部相机会随构型变化 |

## 6. 阶段 1 的产物

| 文件 | 内容 |
|---|---|
| `results/experiment_2_1/frozen_config.json` | 冻结的模型/配置/判据 |
| `results/experiment_2_1/audit/summary.json` | 全部核对数值 |
| `results/experiment_2_1/audit/checks.csv` | 15 条检查，全部通过 |
| `results/experiment_2_1/audit/start_joints.csv` | 480 条 episode 的起始关节角 |
| `scripts/audit_experiment_2_1.py` | 可复跑的核对脚本 |

## 7. 复现

```bash
cd ~/simulate_vla
PYTHONPATH=src venv/bin/python -u scripts/audit_experiment_2_1.py \
  --dataset data/lerobot_panda_multi --diag data/_shards_exp2 \
  --results results/experiment_2 --out results/experiment_2_1/audit
```
