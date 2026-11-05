# 实验 4.3 验收对照表（逐条对账）

对照任务书 §五（质量验收）、§七（执行顺序）、§八（交付清单）逐条给出状态与证据。
状态：**PASS** / **PARTIAL** / **NOT DONE**。

## A. 采集与数量（§二、§七.3-5）

| 要求 | 状态 | 证据 |
| --- | --- | --- |
| A90 新增 70 → train 100 | **PASS（140）** | `train_manifest_v2.json`、`strict_success_results.csv` |
| B90 新增 69 → train 100 | **PASS（113）** | 同上 |
| 只用 train 条件、不碰 val/test | **PASS** | `train_split_manifest.json`（交集为空） |
| 新 seed 与旧 seed 不重复 | **PASS** | `train_manifest_v2.json`（253 唯一 (skill,seed)） |
| 失败样本隔离不入库 | **PASS** | `records/` 保留 58 条失败，数据集不含 |

## B. 物理成功（§五.A）

| 要求 | 状态 | 证据 |
| --- | --- | --- |
| 新增 episode 全部过 `exp4.1-strict-v1.2` | **PASS** | 采集时逐条判定，仅 strict_success 入库 |
| 旧 A90 示范复核 v1.2 | **PASS（30/30）** | `old_data_recheck.md`、`old_v12_recheck.csv` |
| 旧 B90 复核 v1.2 | **PASS（28/31，3 条剔除）** | 同上 + `train_manifest_v3.json` 的 `excluded` |
| 墙接触是真实物理接触 | **PASS** | B90 每条 195–935 个墙接触物理步（`coverage_v2.md`） |

## C. 连续执行与动作一致性（§五.B）

| 要求 | 状态 | 证据 |
| --- | --- | --- |
| 维度/有限值/NaN | **PASS（253/253）** | `preflight_full.json` |
| 时间戳单调、dt=0.05 s | **PASS** | 同上 |
| 视频帧数 = parquet 行数（506 个视频） | **PASS** | 同上 |
| 动作-状态时序（无错帧） | **PASS** | `action_timing_check.json`：\|a−s_next\|<\|a−s_now\| 100%，中位 0.0006 vs 0.0081 rad |
| 图像-状态同步 | **PASS（定量）** | `content_verification.md`：相关 A90 0.455 / B90 0.732（22/22 正） |
| 无意外 reset / teacher forcing | **PASS（结构性）** | 采集器无 reset；episode 为连续 rollout |
| 视频/轨迹抽查（一定比例） | **PARTIAL** | 26 条自动（0 空白）+ 4 张人工查看（A90 top/wrist、B90×2）+ 22 条定量核验；**未逐帧人工审阅** |

## D. 数据覆盖（§五.C）

| 要求 | 状态 | 证据 |
| --- | --- | --- |
| 每个训练条件的独立成功数 | **PASS** | `coverage_v2.md`：A90 16–26、B90 22–36，10/10 条件 |
| A90 阶段窗口（接近/抓取/抬升/搬运/放置） | **PARTIAL（代理）** | `coverage_v2.md`：接近 28.6%、抓取-搬运 42.6%、放置 28.8%（夹爪事件划分）；子阶段细分依赖 records 指标 |
| B90 阶段窗口（接近/推动/墙接触/保持） | **PASS** | `coverage_v2.md`：22.3% / 42.3% / 35.3% |
| 静止帧比例 | **PASS** | A90 31.2%、B90 36.9% |
| 末端-物体相对位姿分布 | **PARTIAL** | 只用 records 的 `ee_object_distance_m` 终点值；未做逐帧相对位姿分布 |
| 接触阶段 EE Δz / 夹爪 / 关节分布 | **NOT DONE** | 未按接触阶段切片统计 |

## E. 训练前 preflight（§五.D、§七.7）

| 要求 | 状态 | 证据 |
| --- | --- | --- |
| 导出到训练接口 | **PASS** | `export_and_probe.md`：`/scratch/lijh/exp43/export_v3`，311 train |
| 全量维度/有限值/帧数/时间顺序 | **PASS** | `preflight_full.json` |
| Anchor-16 标签构造 | **PASS（抽样 11 条）** | `preflight_full.json`：形状/有限/k=0 一致 |
| train-only normalization 诊断（不改 stats） | **PASS** | `train_norm_stats_audit.json`：越界 0，DROID 仍适用 |
| `train_lora.py --mode probe` | **PASS** | `EXPORT LOSS` 见 `final_report.md` §12（2.093963） |
| 原始数据/导出 SHA256 | **PASS** | `selected_artifacts_sha256.json`（253+506）、`export_sha256.json`（761 文件 17.95 GB） |

## F. 交付清单（§八）

| 文件 | 状态 |
| --- | --- |
| `generation_config.json` | ✅ |
| `dataset_manifest.json` | ✅（v2 逐条 + v3 训练清单） |
| `quality_audit.md` | ✅ |
| `strict_success_results.csv` | ✅ |
| `coverage_comparison.json` | ✅（另有 `coverage_v2.json/.md`） |
| `train_split_manifest.json` | ✅（另有 `train_manifest_v2/v3.json`） |
| `preflight_report.md` | ✅ |
| `final_report.md` | ✅ |
| 大工件 SHA256 | ✅ |
| 复现命令 | ✅ `README.md` |

## G. 八问（简答）

1. **新增多少条**：A90 **140**、B90 **113**（去重后，全部严格成功）。
2. **每条件覆盖**：A90 6 个目标 16–26 条；B90 4 个配置 22–36 条；10/10 条件全覆盖。
3. **新增了什么**：B90 每条 195–935 个真实墙接触物理步、推进 0.173–0.184 m；A90 每条完整闭合-抬升-搬运-放置；A90 目标从每目标 ~5 条增到 16–26 条。
4. **无效/重复**：无效 0 入库（58 条失败隔离）；重复 9 条（首批 offset 碰撞）已剔除，v3 再剔除旧 B90 3 条 v1.2 失败项。
5. **原始数据严重问题**：未发现严重同步/标签问题；A90 30/30 通过 v1.2，B90 3 条不达标（已剔除）。
6. **val/test 是否保留**：保留，16/13，且新增未使用其条件（交集为空）。
7. **preflight 是否通过**：数据层 PASS、导出+probe PASS；剩余 `PARTIAL` 项见 D 表（接触阶段切片统计、逐帧人工审阅）。
8. **能否开始训练**：**可以**（读 `export_v3`，311 train）。建议在训练前顺手补"逐帧接触阶段统计"这一项非阻塞工作。
