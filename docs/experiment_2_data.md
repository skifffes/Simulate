# 实验 2 · 阶段 2：原子技能示范采集

## 1. 目标与范围

为四个原子技能各采集 120 条**独立**专家示范，合成一个多任务 LeRobot v2.1 数据集，供共享 SmolVLA 微调：

- `A0`：抓起 → 沿 +Y 搬运 → 保持长轴 Y 放置释放
- `A90`：抓起 → 搬运中真实旋转 90° → 长轴 X 放置释放
- `B`：从物体当前状态沿 +X 推到挡板
- `C`：抓起 →（必要时）转回长轴 Y → 放入托盘释放

**训练集里没有任何 A→B / A→C 连续轨迹**：B/C 的每条 episode 都从静止的独立初始状态开始，物体位置与朝向覆盖 A0/A90 终态附近，但机器人是从 HOME 重新出发的。

## 2. 数据规模

| 技能 | episode | 帧数 | 平均控制周期 | 采集尝试 | 专家成功率 |
|---|---:|---:|---:|---:|---:|
| A0 | 120 | 16 164 | 135 | 120 | 100% |
| A90 | 120 | 21 450 | 179 | 120 | 100% |
| B | 120 | 14 569 | 122 | 176 | **68%** |
| C | 120 | 16 625 | 138 | 122 | 98% |
| **合计** | **480** | **68 808** | — | 538 | — |

- 20 Hz，256×256，两路相机（top + wrist），9 维本体状态，8 维动作。
- 数据集大小 218 MB（含 960 段 H.264 视频）。
- 服务器路径：`~/simulate_vla/data/lerobot_panda_multi`。

## 3. 初始状态分布

| 技能 | 位置（m） | 朝向 | 说明 |
|---|---|---|---|
| A0/A90 | (0.55±0.008, 0±0.008) | 长轴 Y（yaw 0） | 与阶段 1 一致 |
| B/C | (0.55±0.012, 0.18±0.012) | yaw ∈ {0, 90°} ± 0.1 rad，各半 | 覆盖 A0/A90 终态附近的物体状态 |

B/C 的物体朝向分布（成功集）：约 54% 长轴 X（yaw≈90°）、46% 长轴 Y，两端各带 ±0.1 rad 抖动。

## 4. 观测、动作与语言

- 观测：`observation.images.top`、`observation.images.wrist`（256×256 RGB，TINY_RENDERER，`supersample=1`）；`observation.state` = 7 关节角 + 2 指实际开度。
- 动作：7 个关节位置目标 + 夹爪目标开度（单指位移），每个动作保持 12 个物理步（20 Hz）。
- 语言指令（`src/vla/atomic_tasks/config2.py`）：
  - A0：*pick up the orange block and place it in the target area, keeping its long side aligned with Y*
  - A90：*…, with its long side aligned with X*
  - B：*push the orange block along the positive X direction to the wall*
  - C：*pick up the orange block and place it into the tray, with its long side aligned with Y*

录制由 `RecordingExpert` 完成：每个控制周期开始时记录当前观测与**即将执行**的动作，再保持 12 步，语义与实验 1 完全一致。

## 5. 采集过程中修掉的问题

1. **C 的旋转 IK 不可达**：从 HOME 出发时，部分种子在"转回长轴 Y"的 -90° 方向无 IK 解。改为失败时自动尝试反方向 +90°（同样能把长轴转回 Y），C 的成功率从 0% 恢复到 98%。
2. **B 的跟踪误差误判**：推到物体时关节受阻力，跟踪误差超过 0.06 rad 就被判失败。推动段改用 0.25 rad 阈值，最终是否推到位由物体前缘间隙判定；B 的成功率从约 50% 升到 68%。
3. **B 仍然是最脆弱的技能**：176 次尝试才凑齐 120 条成功示范。失败集中在推进最后一段（`b_push_6`，34 次）和起始接触段（`b_push_2`，24 次），说明"贴上去"和"推到墙边"两个接触瞬间对控制周期较敏感——这会原样传给 VLA，是后面重点观察的对象。

## 6. 数据校验

`scripts/validate_atomic_dataset.py` 对全部 480 集做了 11 540 项检查，**0 失败**：

- 元数据、逐集长度、时间戳步长、`frame_index` / `episode_index` / 全局 `index` 连续；
- 关节限位、夹爪范围、动作跳变、动作指向下一周期观测（相关系数）；
- `episodes_stats` 与实算统计一致；
- 每集两路视频的帧数、分辨率、时长与 parquet 对齐。

动作回放（2 集）：平均关节偏差 **6.6×10⁻⁶ rad**，最大 2.6×10⁻³ rad，说明"动作-状态"语义闭环自洽。

校验产物在服务器 `results/experiment_2/validation/`（`report.json`、`checks.csv`、抽帧拼图、关节曲线）。

## 7. 终态精度（专家自身）

| 技能 | 指标 | 均值 ± 标准差 |
|---|---|---|
| B | 前缘距挡板 | 7.8 ± 3.6 mm（最大 17.6 mm） |
| B | 侧向偏移 | 3.7 ± 3.6 mm |
| B | yaw 变化 | 0.013 ± 0.074 rad（最大 0.26 rad） |
| C | 托盘中心 x 误差 | −2.4 ± 0.7 mm |
| C | 托盘中心 y 误差 | −1.2 ± 0.4 mm |

## 8. 训练/测试隔离

- 采集种子：A0 `900000+`、A90 `910000+`、B `920000+`、C `930000+`。
- 后续评测使用完全不同的种子段（建议 `40000000+`），且组合评测的初始场景在四个组合间配对。
- 验证/测试按 episode 与种子隔离，训练集中不会出现评测用初始状态。

## 9. 文件

| 位置 | 内容 |
|---|---|
| `src/vla/atomic_tasks/config2.py` | 相机参数、四个任务的语言指令、特征定义 |
| `src/vla/atomic_tasks/recording.py` | 20 Hz 录制版专家（A0/A90/B/C 通用） |
| `src/vla/lerobot_writer.py` | 新增 `merge_multi_task`：按 episode 保留任务文本 |
| `scripts/collect_multi_atomic_data.py` | 单技能采集 + 合并 |
| 服务器 `data/lerobot_panda_multi` | 480 集多任务数据集 |
| 服务器 `results/experiment_2/validation` | 校验报告与抽查图 |

## 10. 复现

```bash
# 单技能采集（服务器）
cd ~/simulate_vla
PYTHONPATH=. venv/bin/python -u scripts/collect_multi_atomic_data.py \
  --skill A0 --episodes 120 --seed-base 900000 --out data/_shards_exp2/A0 --overwrite

# 合并
PYTHONPATH=. venv/bin/python -u scripts/collect_multi_atomic_data.py \
  --merge --out data/lerobot_panda_multi --shards data/_shards_exp2

# 校验
PYTHONPATH=$PWD/src venv/bin/python -u scripts/validate_atomic_dataset.py \
  --dataset data/lerobot_panda_multi --out results/experiment_2/validation \
  --samples 4 --replay 2
```

## 11. 下一步（阶段 3）

1. 用该数据集微调**一个共享 SmolVLA**（4 个任务，语言区分），先跑通 batch/forward/backward 与小步数冒烟。
2. 正式训练在单张空闲 A40 上进行，按原子验证集选择 checkpoint，**不使用任何组合成功率选模型**。
3. 训练完成后先做四类独立评测（各 ≥20 集），再进入组合实验。
