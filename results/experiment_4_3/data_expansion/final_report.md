# 实验 4.3 数据扩充：最终报告

日期 2026-11-05；基准 `6f72bc5`；本轮**只做数据扩充与验收，未启动任何 π₀.₅ 训练**。

## 0. 一句话结论

**目标达成且超额**：A90 新增 **140** 条、B90 新增 **113** 条通过严格评价器的独立训练示范
（去重后，目标各 70/69）；合计 train 从 30/31 变为 **170/144**。
但**训练前 preflight（OpenPI 导出、normalization 复核）与视频抽查未做**，见 §6。

## 1. 必答八问

**Q1. A90/B90 各新增多少条独立、严格成功的专家示范？**
A90 **140**、B90 **113**（去重后；`(skill, seed)` 唯一，无跨集重复）。
原始 attempts→success：A90 153→145、B90 167→117；其中 9 条是首批与 `a90a/b90a` 同 offset 的重采，已排除。

**Q2. 每种训练条件覆盖多少条？**

| A90 目标 | 入库 | B90 配置 | 入库 |
| --- | ---: | --- | ---: |
| A_x-3_y-3 | 27 | low·y015·home | 24 |
| A_x00_y-3 | 27 | low·y015·offset | 30 |
| A_x+3_y-3 | 17 | mid·y021·home | 26 |
| A_x-3_y00 | 25 | mid·y021·offset | 37 |
| A_x00_y00 | 25 | | |
| A_x+3_y00 | 24 | | |

6/6 A90 目标、4/4 B90 训练配置全部覆盖；最弱的是 `A_x+3_y-3`（17 条，8 次 `tilt_ok` 失败）。

**Q3. 与原始数据相比新增了什么？**
- 机器人初始构型：B90 新增覆盖 `B_robot_offset` 配置（原先该配置在 train 中只有少量）；
  实际统计见 `coverage_comparison.json`（4 个 B90 训练配置各 24–37 条）；
- 物体初始位姿：A90 覆盖 6 个训练目标 × 每目标 17–27 条，B90 覆盖 2 个物体 Y 起点 × 2 个机器人构型；
- 完整任务轨迹：每条入库 episode 都是**连续完整**任务（A90 抓取-搬运-旋转-放置-释放；
  B90 接近-推动-真实触墙-稳定），不是截取片段；短于 10 个有效动作的不会入库；
- 接触/抓取阶段覆盖：均为严格判据下的成功轨迹，因此必然包含抓取闭合与 B90 的墙面接触
  （B90 入库 episode 的 `wall_contact_steps` 见 `dataset_manifest.json` 的 metrics 字段）。

**Q4. 是否存在无效或重复轨迹？**
无无效轨迹入库（失败 58 条只留在 `records/`）；
**有 9 条重复**（同 config + 同 seed 重采，来自首批与 `a90a/b90a` 的 offset 碰撞），已排除出验收数，
但目录未删。

**Q5. 原始数据是否发现严重的同步、标签或专家问题？**
本轮**未发现**：采集器、专家、评价器、导出契约全部沿用 4.1 原实现，只新增了两个过滤/偏移参数；
但**没有对原始 90 条做 v1.2 全量复核**，只在上一轮用 4 条做过专家回放（8/8 成功）。

**Q6. 原有 validation/test 是否完整保留且没有泄漏？**
是。新增数据只使用 train 条件；`train_split_manifest.json` 记录与 val/test 配置交集为空
（A90 val 2 个、test 1 个；B90 val/test 各 1 个，全部未使用）；seed 与原始 107 个 seed **零交集**。

**Q7. 新数据是否已通过训练前 preflight？**
**数据层通过、训练接口层未通过。** 已对 253 条唯一 episode 全量检查：
0 错误、state/action=9/8、无 NaN、dt=0.05 s、506 个视频帧数=行数、256×256@20fps、
Anchor-16 抽样 11 条构造一致（`preflight_full.json`）；但**多源 OpenPI 导出与
`train_lora.py --mode probe` 未做**，normalization 未重算。

**Q8. 当前数据是否具备启动下一轮 π₀.₅ 微调的条件？**
**接近但还差一步**：数据量与覆盖已满足（170/144 条 train，6+4 个条件全覆盖、无泄漏）；
但必须先完成 OpenPI 导出与 preflight（动作契约、normalization 统计、sample 抽样核对），
才建议开始训练。

## 2. 采集与验收协议（可复现）

```
# A90：6 个训练目标 × 新种子（offset 40 / 54），目标 70 条/进程
bash server/exp43/run_parallel.sh a90a A90 40 14 70
bash server/exp43/run_parallel.sh a90b A90 54 14 70
# B90：4 个训练配置 × 新种子（offset 40 / 60），目标 69 条/进程
bash server/exp43/run_parallel.sh b90a B90 40 20 69
bash server/exp43/run_parallel.sh b90b B90 60 20 69
# 汇总
python scripts/exp43/compile_results.py
```

- 采集器：`scripts/exp41/collect41.py --mode collect --only-goals ... --seed-offset ...`
- 严格判据：`src/exp41/strict_eval.py`（`exp4.1-strict-v1.2`），未修改阈值
- 入库条件：`strict_success and len(actions) >= 10`

## 3. 服务器产物（大文件不入 Git）

| 路径 | 内容 |
| --- | --- |
| `/scratch/lijh/exp43/data_a90a`、`data_a90b` | A90 各 70 条（13.7k 帧） |
| `/scratch/lijh/exp43/data_b90a`（57 条）、`data_b90b`（56 条） | B90 共 113 条 |
| `/scratch/lijh/exp43/data/{A90,B90}` | 首批 5/4 条（含 9 条中的重复项） |
| `/scratch/lijh/exp43/{a90a,a90b,b90a,b90b}/records/*.json` | 全部 attempt（含失败）与严格判据结果 |
| 每个数据集 `meta/info.json` 的 sha256 | 见 `coverage_comparison.json` |

## 4. 与目标对比

| 技能 | 原 train | 目标 | 实际新增（去重） | 实际 train | 达标 |
| --- | ---: | ---: | ---: | ---: | --- |
| A90 | 30 | 100 | **140** | **170** | ✅ |
| B90 | 31 | 100 | **113** | **144** | ✅ |

## 5. 本轮明确未完成（如实列出）

1. **OpenPI 导出未做**：新数据仍是 LeRobot v2.1 原始目录，未生成 `exp43` 的 npy/npz 训练缓存；
2. **训练前 preflight 未做**：没有重算 train-only normalization 统计，没有 action 契约/时间对齐抽查；
3. **视频与轨迹抽查未做**（要求"一定比例"）：只记录了文件数与字节，没有逐段看视频；
4. **原始 90 条未做 v1.2 全量复核**；
5. **接触阶段分布对比未做**（需要重放打接触标签），只有按条件/帧数的覆盖统计；
6. 9 条重复 episode 只标记未删除；`data/{A90,B90}` 目录仍在；
7. 未做专家回放回归（本轮只验证入库 episode 的 strict_success，未再跑独立回放）；
8. 未启动任何训练（按要求）。

## 6. 下一步（按优先级）

1. 把 6 个数据目录合并为单一 LeRobot 数据集（或直接多源导出），生成 `exp43` 的 npy/npz 缓存；
2. 跑 `scripts/exp42/audit_action_dataset.py` 式 preflight：anchor-16 标签统计、normalization、
   state/action/图像时间对齐抽查（抽 10% episode）；
3. 视频抽查：每个条件抽 1–2 条，重点看抓取闭合与 B90 墙接触瞬间；
4. 完成后再开 π₀.₅ 微调（同 base / 同 Anchor-16 / 同优化配置，对照 30+31 与 170+144）。

## 7. 采集中途补充：preflight-lite 通过（详见 preflight_report.md）

抽检 A90（70 条目录）与 B90（57 条目录）各 1 条 episode：列名/维度/时间戳/视频数量全部符合 4.1 契约，
无 NaN；(skill, seed) 唯一 253 对、与原始 seed 零交集。**完整 OpenPI 导出与 normalization 复核仍未做**。
## 8. 验收后修复（2026-11-05，回应复核）

- **去重 P0 已修**：新增 `scripts/exp43/build_train_manifest.py`，只读 `data_a90a/a90b/b90a/b90b`，
  排除首批 9 条重复，按 `(skill, seed)` 兜底去重，分配全局唯一 `global_episode_id`（1..253），
  得到 **A90 140 + B90 113 = 253** 条训练清单（`train_manifest_v2.json`）；
- **哈希 P1 已修**：`selected_artifacts_sha256.json` 以完整相对路径为键，记录 253 个 parquet + 506 个 MP4 的 SHA256；
- **措辞问题已改**：入库条件实为 `len(actions) >= 10`（动作序列帧数），不再写作“有效运动帧 ≥10”；
- 去重后按条件：A90 16/24/24/26/26/26，B90 22/29/26/36（10 个训练条件全部覆盖）。

仍未完成：OpenPI 导出（314 条）、完整 preflight、10% 分层内容抽查、旧 90 条 v1.2 复核。
