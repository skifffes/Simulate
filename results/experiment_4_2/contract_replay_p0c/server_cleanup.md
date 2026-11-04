# 服务器清理与脚本安全（2026-11-04）

## 1. `/scratch` 现状

| 时点 | 可用 | 说明 |
| --- | --- | --- |
| 清理前 | 57 GB | 多份 12.6 GB 合并参数 + 两个旧项目 |
| 删除 `OneTwoVLA`（7.5 GB）+ `DINO-WM`（26 GB）后 | 90 GB | 用户确认的旧项目 |
| 删除**可再生的合并参数**后 | **166 GB** | 见下表 |

`du -sh /scratch/lijh/exp42/*`（清理前）：`ab_repr` 45G、`curve1200` 36G、`curve600` 33G、
`ab` 30G、`run` 28G、`run2` 17G、`overfit` 5.9G、`data` 5.9G。

## 2. 删除了什么（全部可由保留的 LoRA npz 重新导出）

| 路径 | 大小 | 可再生来源（已保留） |
| --- | --- | --- |
| `ab_repr/A/ckpt_merged` | 12 G | `ab_repr/A/ckpts/lora_step00300.npz` |
| `ab_repr/B/ckpt_merged` | 12 G | `ab_repr/B/ckpts/lora_step00300.npz` |
| `ab_repr/A/ckpts/params_step00300` | 5.9 G | 同上 |
| `ab_repr/B/ckpts/params_step00300` | 5.9 G | 同上 |
| `curve600/ckpt_300` | 12 G | `curve600/ckpts/lora_step00300.npz` |
| `curve600/ckpt_600` | 12 G | `curve600/ckpts/lora_step00600.npz` |
| `curve600/ckpts/params_step00600` | 5.9 G | 同上 |
| `curve1200/ckpt_900` | 12 G | `curve1200/ckpts/lora_step00900.npz` |

执行脚本 `server/exp42/safe_cleanup_merged.sh`：逐项检查「LoRA 源存在 + 属主是自己」才删，
不使用通配符，删完自检保留项。

**保留并已自检存在**：
`curve1200/ckpt_1200/params/_METADATA`、`curve1200/ckpts/lora_step01200.npz`、
`run2/ckpts/params_nested`（step901）、`overfit/lr4/params_step00150`、`exp42/data/meta.jsonl`、
所有 `lora_step*.npz`、`assets/droid/norm_stats.json`。

## 3. 脚本安全修正（审计指出的问题）

`server/exp42/cleanup_gpu.sh` 直接对四个**固定 PID** 执行 `kill`，没有校验属主。
在共用服务器上 PID 会被复用，可能误杀别人的作业。已删除该脚本，替换为
`server/exp42/kill_gpu_procs_safe.sh`：

- 默认 **dry-run**：只列出 GPU 进程的 pid / uid / 命令；
- `--apply` 必须同时给 `--pattern`，且只 kill **属于当前 uid 且命令行匹配** 的进程；
- 实测：本机 uid=1072（lijh）匹配到自己的训练进程，另外 4 个 uid=1085（他人）被标为 `other-user` 且不会被选中。
