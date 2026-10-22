# 实验 1.1 结果目录

严格自主评测 + 受控 OOD 的原始记录与汇总。报告见 [docs/experiment_1_1.md](../../docs/experiment_1_1.md)。

## 看什么

| 文件 | 内容 |
|---|---|
| `summary.md` / `summary.csv` | 15 组成功率、Wilson 区间、平均抬升与放置误差 |
| `episodes_all.csv` | 300 条 episode 逐集指标（含阶段事件、闭爪偏差、双指接触、失败主因） |
| `paired_tests.csv` | 同种子配对的 McNemar 检验 |
| `failures_primary.csv` | 失败主因计数 |
| `success_rate.png` | 成功率条形图（95% 区间） |
| `failure_primary.png` | 失败原因分布 |
| `render_check/` | 采集 vs 评测渲染一致性（MAE/PSNR/对比图） |
| `config.json` | checkpoint、种子、各组的 mode/supersample/OOD 参数 |
| `*/eval.json` | 每组原始记录（逐集诊断字段） |
| `strict_chunk50_iid20/episodes/videos/` | 5 条代表性视频（2 成功 + 3 失败） |

## 口径

- `strict`：判定前不替模型张爪、不修正末端、不动工件；必须出现模型自己的张开指令才算释放。
- `assisted`：旧口径，仅用于解释历史数据。
- 所有新评测 `supersample=1`（与数据采集一致）；`strict_n10_ss2_iid20` 是专门的渲染对照。
- 种子 30000000 起是 **IID held-out**，不是 OOD；真正的 OOD 只有 shift-x / yaw / friction 三组。
