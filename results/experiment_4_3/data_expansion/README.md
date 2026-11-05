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
