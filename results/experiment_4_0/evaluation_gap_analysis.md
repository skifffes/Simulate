# 实验 4.0 附：B90 评价接口缺口与 4.1 的校准需求

本文件只做**静态审计**，不重建评价器、不收集新示范、不计算正式成功率（prompt §九）。
证据全部来自现有代码，行号为当前 HEAD（`e048313`）。

## 1. 现有 B90 判据的实际代码

`scripts/eval_multi_atomic_vla.py:120-128`：

```python
elif skill == 'B':
    depth = BLOCK_HALF[0] * abs(math.cos(yaw)) + BLOCK_HALF[1] * abs(math.sin(yaw))
    gap = float(WALL_INNER_X - (pos[0] + depth))
    dy = float(pos[1] - initial[1])
    dyaw = abs(math.atan2(math.sin(yaw - initial[3]), math.cos(yaw - initial[3])))
    ok = (abs(gap) <= 0.02 and abs(dy) <= 0.02 and dyaw <= math.radians(15)
          and tilt <= 12 and rest)
```

配套量（`:92-108`）：
- `tilt = |roll| + |pitch|`（度，逐轴取 wrap 后的绝对值再相加）；
- `rest = speed < 0.01`（`p.getBaseVelocity` 的瞬时线速度）；
- `WALL_INNER_X = 0.78`（`src/vla/atomic_tasks/scene2.py`），`BLOCK_HALF = (0.02, 0.05, 0.02)`。

## 2. 具体缺口（逐条给出会被误判的情形）

| # | 缺口 | 代码位置 | 后果 |
| --- | --- | --- | --- |
| G1 | 只用**几何间隙**判定，从未查询 block↔wall 的真实接触 | 全仓搜索 `getContactPoints` 未见 `bodyB=wall` 的用法；`sim.collisions()` 只查机器人与环境（`src/simulation.py:96-118`） | “离墙很近”被当成“推到墙”，实验 3.5 的 87/105 近墙成功里 0 次真实接触 |
| G2 | `abs(gap) <= 0.02` 允许 `gap < 0`（物体侵入墙体） | `:125` | 穿模/挤压被算成功；而 3.2 的 B90 实验用 `0.0 <= gap`（`results/experiment_3_2/wall_contact_audit.csv`），**同仓两套不等价判据** |
| G3 | 没有接触**法向**与**法向力** | 无 | 无法区分“沿 +X 顶住墙”与“侧向蹭到墙” |
| G4 | 没有记录接触 body/link id | 无 | 无法排除“机器人自身或托盘碰到墙”被当成任务接触 |
| G5 | 不区分**初始接触**与**执行后接触** | 无 | 若初始状态就贴墙，0 动作也能“成功” |
| G6 | 只查净位移 `|dy|`，不查路径 | `:123` | 先侧向拖开再推回墙边也可通过 |
| G7 | `rest` 用瞬时速度、无静置窗口 | `:104` | 抖动中的瞬间静止即通过（3.2 阶段在外部补了 1.0 s 静置，但没进这个函数） |
| G8 | `tilt <= 12°` 与 `gap <= 2 cm` 无专家成功/失败校准记录 | 常量硬编码 | 阈值来源不明，不能作为冻结判据 |
| G9 | `depth` 用了绕 Z 的 yaw 近似投影，物体若绕 X/Y 翻转（tilt 大时）投影失效 | `:121` | tilt 与 depth 同时异常时判据自相矛盾 |

## 3. 4.1 需要新增/修改的评价项（建议，不在 4.0 实现）

1. **真实接触**：`p.getContactPoints(bodyA=block, bodyB=wall)`，记录
   `(linkIndexA/B, contactNormalOnB, normalForce, lateralFriction1/2, contactDistance)` 的**时间序列**；
2. **方向性**：接触法向与 +X 的夹角需在阈值内（例如 ≤ 20°），且法向力需超过
   `max(0.05 N, 5% 物体重力)`，避免“擦到”；
3. **非穿透**：要求 `gap >= -1 mm`（数值容差），与接触判据同时成立才算“推墙”；
4. **静置窗口**：连续 0.5–1.0 s 内 `speed < 0.01 m/s` 且无新接触事件；
5. **路径**：记录 `x(t)` 单调段占比，或要求 +X 方向累计正位移 ≥ 最终净位移的 80%；
6. **初始接触排除**：t=0 已接触 wall 的回合应单独标记并排除；
7. **阈值校准**：用 4.1 采集的专家成功/失败各 ≥20 例，取使 Youden J 最大的阈值，
   冻结后写入 `config.yaml` 并加版本号；
8. **一致性**：全仓只保留**一套** B90 判据（消除 G2 的 `abs(gap)` vs `0.0 <= gap` 分裂）。

## 4. 建议的验证测试（4.1 执行）

| 测试 | 目的 | 通过条件 |
| --- | --- | --- |
| 静止贴墙场景 | 初始已接触不应算成功 | 判据输出 `initial_contact=True` 且 success=False |
| 已知好的推墙回合（专家） | 校准 gap/接触/力阈值 | 专家成功例 ≥90% 判成功 |
| 只在墙前 1.5 cm 停住 | 近墙 ≠ 推墙 | 判失败（gap>2mm 或无接触） |
| 侧向经过墙边 | 方向判据有效 | 法向夹角判失败 |
| 机器人蹭墙 | link 白名单有效 | 非 block-wall 接触不算任务接触 |
| 物体翻转 | tilt/depth 一致 | tilt>12° 必失败且 depth 计算不报错 |
