# 实验 4.3 训练前 preflight（**部分完成**，如实标注）

日期 2026-11-05。本轮时间窗内**只完成了 preflight-lite**；完整的 OpenPI 导出与
train-only normalization 复核**未做**（见 §3）。

## 1. 已做：preflight-lite（每条技能抽 1 条）

| 检查 | A90（data_a90a，70 条） | B90（data_b90a，57 条） |
| --- | --- | --- |
| parquet 列 | `observation.state, action, timestamp, frame_index, episode_index, index, task_index` | 同 |
| 抽检 episode 行数 | 198 | 157 |
| state / action 形状 | (198, 9) / (198, 8) | (157, 9) / (157, 8) |
| NaN | 无 | 无 |
| 时间戳单调 | 是 | 是 |
| 视频文件数（期望 = 2×episodes） | 140 = 2×70 ✓ | 114 = 2×57 ✓ |
| 夹爪动作范围 | 见 episode（0.015–0.04 之间变化） | 0.014 恒定（B90 全程闭合） |
| \|Δq\| 量级 | 与 4.1 记录同量级 | max 0.0415 rad |

结论：**新数据的文件结构、维度、时间戳与 4.1 契约一致**，没有 NaN、没有视频缺失。

## 2. 已做：隔离与去重

- `train_split_manifest.json`：新增数据使用的配置与 val/test 配置交集为空；
- seed 与原始 107 个 seed **零交集**；
- `(skill, seed)` 唯一 253 对（9 条首批重采已排除出验收数）。

## 3. 未做（明确列出）

1. **OpenPI/LeRobot 训练缓存导出**：未生成 `exp43` 的 npy/npz（Anchor-16 标签、图像数组）；
2. **train-only normalization 统计**：未按新数据重算 Δq 分布、未与 DROID stats 对比（按任务要求，
   本轮也不允许擅自改 normalization）；
3. **完整 action/observation 契约抽查**：只抽了 2 条 episode 的结构检查，
   未做逐帧 state/action 滞后、图像-状态同步的交叉验证；
4. **视频内容抽查**：只统计了 mp4 数量，未逐段查看；
5. **原始 90 条的 v1.2 复核**：未做；
6. **专家回放回归**：未做（本轮仅依赖入库时的严格判据）。
