# 实验 0C 第一阶段结果

由 `scripts/run_experiment_0c.py --episodes 5` 生成。工件被真实夹爪抓起并搬运，未使用瞬移或固定约束。

## 文件

| 文件 | 内容 |
|---|---|
| `result.json` | 第一次 episode 的全部判定指标、阶段耗时、参数与失败码 |
| `summary.json` | 5 次重复的汇总（均值/标准差/极值） |
| `repeatability.csv` | 逐次一行，含抬升、滑移、放置误差、终态速度、闭合指宽 |
| `trajectory.csv` | 工件与末端逐步轨迹（每 4 个仿真步），含接触力、倾角 |
| `contacts.csv` | 工件与机械臂的接触点记录（每 8 步），含 link、法向力、接触位置 |
| `grasp.gif` | 完整抓放过程回放，20 fps |
| `keyframes.png` | 各阶段关键帧拼图 |
| `keyframes/` | 各阶段首帧原图（PNG） |
| `frames/` | GIF 的逐帧 PNG |

## 本次结果

- 5/5 成功，失败码为空。
- 抬升 0.1000 m，双指接触力 20.00 / 20.00 N。
- 相对滑移 3.45 mm，放置 xy 误差 2.83 mm，高度误差 −0.01 mm。
- 终态倾角 0.0003°，线速度 1.2e-10 m/s，释放后接触点数 0。

5 次逐位相同，因为固定场景 + 确定性求解器。**这是可重复性检查，不是独立物理试验，也不是鲁棒成功率。**

## 相邻数据

- `../experiment_0c_jitter/`：工件起点随机平移 ±4 mm、绕竖轴 ±0.1 rad，5/5 成功。
- `../experiment_0c_diagnose/`：摩擦 × 夹持力 16 组负对照，9/16 成功，失败均为 `object_not_lifted`。

设计与分析见 [docs/experiment_0c_stage1.md](../../docs/experiment_0c_stage1.md)。
