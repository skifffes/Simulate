# 实验 2.8-S1 修订记录

修订日期：2026-10-30  
代码：`scripts/exp28s1_target_height.py`、`scripts/exp28s1_visualize.py`

## 1. 修复的代码错误

### 1.1 horizon 分组索引错配

原代码：

```python
for i in np.where(m)[0]:
    chunk[i]['dz_target_mm']
```

`i` 是逐帧 `dyn` 的索引，而 `chunk` 每帧有 5 条 horizon 记录，导致帧与 horizon 错位。

修复：

```python
chunk_map = {(int(c['episode_id']), int(c['frame_id']), int(c['horizon'])): c
             for c in chunk}
```

按 `(episode, frame, horizon)` 精确查找。

**验证：**

- `gripper_target_statistics.csv` 中每组 `dz_target_h0_mean_mm` 现在严格等于 `dz_target0_mean_mm`。
- 例：35–40 mm 组修复前 h0 = −3.49 mm、直接值 0 mm；修复后两者均为 **0.0 mm**。
- `expert_chunk_targets.csv` 共 **35,080 行 = 7,016 帧 ×5 horizons**，行数未变；只是汇总读取方式修正。

### 1.2 35 mm 浮点边界

原边界：`(25, 35)` 与 `(35, 40.5)`，34.99998 与 35.00002 被拆开。

修复：

- 14：13.0–15.5
- 16–25：15.5–24.5
- 25–35：24.5–34.5
- 35–40：34.5–40.5

**数值变化：**

| 组 | 修复前 | 修复后 |
|---|---:|---:|
| 25–35 mm | 192 帧 | **180 帧** |
| 35–40 mm | 48 帧 / 48 eps | **60 帧 / 60 eps** |

Figure 1 的红色 bin mean 与统计 CSV 现在使用同一组边界。

### 1.3 estimable 逻辑

原逻辑用逐渐增长的 `avail`，导致第一个组被误判 False、后续组仅因“已经过两个组”被误判 True。

修复为两阶段：

1. 先收集同一 progress 区间的所有夹爪组；
2. 仅当本组和至少一个其他组都满足 `n_frames ≥ 50` 且 `n_episodes ≥ 10` 时，本组 `estimable=True`；
3. 增加 `n_qualifying_grip_groups`。

**结果：**

- 0–15%：4 个组均满足，`estimable=True`；
- 15–30%、30–50%：只有 14 mm，`False`；
- 50–100%：只有 14 mm；16–25 的 19 帧 / 4 episodes 为 `False`。

注意：`estimable` 只表示最低样本覆盖门槛，**不代表固定构型下存在因果可识别的 14→40 对照**。即使 0–15% 为 True，四个组仍处于不同时间点/EE 高度，不是固定七关节构型。

## 2. 报告表述修正

### 2.1 h0/h4 与 h9/h19/h49

修正后：

| 夹爪 | h0 | h4 | h9 | h19 | h49 |
|---|---:|---:|---:|---:|---:|
| 14 mm | −3.46 | −17.2 | −34.9 | −65.6 | −122.9 |
| 16–25 | −2.81 | −39.8 | −94.0 | −208.3 | −391.9 |
| 25–35 | 0.00 | −10.8 | −67.9 | −193.0 | −429.8 |
| 35–40 | 0.00 | 0.0 | −44.4 | −167.4 | −429.5 |

单位：mm。

- **h0/h4**：宽夹爪更少向下；
- **h9 起**：wider 组反而更向下；
- **不能使用“h≥4 反转”的概括**，因为 h4 仍与 h0 同向。

### 2.2 首次下降时间

原报告写“目标下降首次在 0.30 s（60/60）”，不准确。

修正：

- 目标下降事件：**51/60 在 0.30 s**；其余 9 条为 0.35、0.40（2 条）、1.05、1.35、1.70（2 条）、1.85、1.95 s；
- 实际下降事件：**59/60 在 0.30 s**；ep96 为 0.70 s；
- 闭合→目标下降延迟仍为 60/60 正，中位 **0.25 s**，范围 0.25–1.90 s；
- ep66 的 1.90 s 延迟保留为异常案例。

### 2.3 结论变化

- Q1：**No（未来目标）/ Mixed（h0）**。未来 horizon 的 wider 组并不更高，反而更低。
- Q2：**Mixed（h0/h4）**。h0/h4 方向一致，h9 起反转。
- Q5：**Mixed（h0）/ No（未来）**。模型“宽→更高”与专家 h9/h19 方向相反；训练数据不能解释模型未来目标。
- “训练数据提供高度响应候选解释”进一步减弱：只能支持早期时序结构，不能支持未来目标方向。

## 3. 验证命令

```bash
# 服务器：重算
HF_HUB_OFFLINE=1 PYTHONPATH=src venv/bin/python scripts/exp28s1_target_height.py

# 检查 h0 与直接值一致
python - <<'PY'
import csv
rows=list(csv.DictReader(open('results/experiment_2_8/supplement_s1/data/gripper_target_statistics.csv')))
for r in rows:
    print(r['grip_bin'], r['dz_target_h0_mean_mm'], r['dz_target0_mean_mm'])
PY

# 检查事件计数
python - <<'PY'
import csv, collections
rows=list(csv.DictReader(open('results/experiment_2_8/supplement_s1/data/episode_timing_events.csv')))
print(collections.Counter(r['target_descent_mid_t'] for r in rows))
print(collections.Counter(r['actual_descent_mid_t'] for r in rows))
PY
```

## 4. 未变化的文件/结论

- `expert_target_dynamics.csv`（逐帧 q/grip/z/Δz/vz）未改变。
- `expert_chunk_targets.csv` 行数/内容未改变；只是分箱汇总读取修正。
- 夹爪闭合时序的“闭合先于下降”方向未变；具体时刻描述已修正。
- 实验 2.8 的冻结模型敏感性与局部覆盖结论未受影响。
- 实验 2.9 A/B 不使用本修订前的 h9/h19 结论作为依据。
