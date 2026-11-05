# 实验 4.3 数据扩充（2026-11-05）

原始 90 条原子示范 → 本轮新增 **A90 140 条 / B90 113 条**（去重后，均通过 `exp4.1-strict-v1.2`），
train 合计 **A90 170 / B90 144**。

| 文件 | 内容 |
| --- | --- |
| `final_report.md` | **最终报告**：八问回答 + 协议 + 未完成清单 |
| `quality_audit.md` | 采集方式、入库规则、失败样本、去重、覆盖、已知局限 |
| `generation_config.json` | 生成配置：split 约束、seed 策略、并行方式、目标 |
| `dataset_manifest.json` | 逐 episode：tag/episode/seed/config/帧数/metrics/data_dir |
| `strict_success_results.csv` | 逐 attempt：skill/config/seed/成功/失败原因/步数 |
| `coverage_comparison.json` | 按条件计数、失败原因统计、数据目录帧数与 info.json sha256 |
| `train_split_manifest.json` | train 条件、seed 去重与"未用 val/test"证明 |
| `large_artifacts_sha256.json` | 服务器大工件（parquet/mp4 计数与字节数）的路径索引 |


| `dedup_fix.md` | **P0/P1 修复**：只读四个无重复目录、全局唯一 episode id、253 parquet + 506 mp4 的 SHA256 |
| `train_manifest_v2.json` | 训练输入清单（253 条唯一 episode，含 provenance 与文件路径） |
| `selected_artifacts_sha256.json` | 入选 parquet/MP4 的逐文件 SHA256 |


| `export_and_probe.md` | **多源导出 + probe**：314 条 train（A90 170/B90 144）、新 episode id ≥1000、forward loss 2.1367 |
| `export_summary.json` | 导出统计（train/val/test 计数与按技能 train 计数） |


| `coverage_v2.md` | **阶段覆盖 v2**：A90 280 帧/条·接近 29%/抓取搬运 43%/放置 29%；B90 推动 42%/墙保持 35%、墙接触 195–935 步 |
| `old_data_recheck.md` | 旧 61 条 v1.2 复核：A90 **30/30**、B90 **28/31**（3 条需剔除） |
| `video_spotcheck.md` | 26 条视频抽查（10.3%）：0 空白、帧间差异 6.7–13.5；人工看过 A90/B90 各 1 张 |
| `old_v12_recheck.csv` | 旧逐条复核结果（含 3 条失败） |
| `export_sha256.json` | 导出 761 个文件（17.95 GB）的 SHA256 |
| `montage_A90_gid001.png` / `montage_B90_gid161.png` | 抽查用三帧拼接图 |


| `train_manifest_v3.json` | **训练输入清单 v3**：311 条（A90 170 / B90 141），已剔除旧 B90 的 3 条 v1.2 失败项 |
| `action_timing_check.json` | 动作-状态时序：\|a−s_next\|<\|a−s_now\| 占 100%，中位 0.0006 vs 0.0081 rad |
| `montage_A90_gid002_top.png` / `montage_A90_gid002_wrist.png` | 第 10 个条件的 top/wrist 抽查图 |


| `content_verification.md` | **视频级定量核验**：图像-关节运动相关 A90 0.455 / B90 0.732；B90 方块像素质心位移 36.6 px ↔ 0.178 m（204 px/m） |
| `image_sync_check.json` | 上述 22 条抽样的逐条数值 |
| `export_v3_summary.json` | 训练就绪数据根 `/scratch/lijh/exp43/export_v3`：311 train / 16 val / 13 test |
| `train_manifest_v3.json` | 训练输入清单 v3（311 条，含 provenance 与剔除原因） |

## 复现

```bash
# 采集（服务器，4 进程并行；每个进程独立 out/data 与 seed 区间）
bash server/exp43/run_parallel.sh a90a A90 40 14 70
bash server/exp43/run_parallel.sh a90b A90 54 14 70
bash server/exp43/run_parallel.sh b90a B90 40 20 69
bash server/exp43/run_parallel.sh b90b B90 60 20 69
# 汇总
python scripts/exp43/compile_results.py
```

## 关键数字

| 指标 | A90 | B90 |
| --- | ---: | ---: |
| 原始 train | 30 | 31 |
| 新增（去重） | 140 | 113 |
| 合计 train | 170 | 144 |
| 尝试 / 成功 | 153 / 145 | 167 / 117 |
| 训练条件覆盖 | 6/6 | 4/4 |

## 未做（详见 `final_report.md` §5）

OpenPI 导出、训练前 preflight、视频/轨迹抽查、原始 90 条的 v1.2 全量复核、接触阶段分布对比。
