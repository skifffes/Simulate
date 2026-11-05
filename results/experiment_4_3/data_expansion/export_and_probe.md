# 多源导出与 probe（2026-11-05 14:35）

## 1. 导出（`scripts/exp43/export_multi.py`）

- 旧 train 61 条：以**符号链接**复用 `/scratch/lijh/exp42/data` 的文件，不复制、不覆盖；
- 新 253 条：从 4 个 LeRobot 目录解码 parquet + 两路 MP4，写成与旧格式一致的
  `ep{idx}_sa.npz` / `ep{idx}_top.npy` / `ep{idx}_wrist.npy`；
- 新 episode 使用 **1000+ 的全局唯一 episode_index**（`A90: 1000..1139`、`B90: 1000..1112`），
  因此不会与旧 `ep000` 冲突；
- val/test 行原样保留（16 + 13），文件同样用符号链接。

| 指标 | 值 |
| --- | ---: |
| 旧 train | 61 |
| 新 train | 253 |
| **train 合计** | **314**（A90 170 / B90 144） |
| val / test | 16 / 13 |
| meta 总行数 | 343 |
| 导出目录 | `/scratch/lijh/exp43/export`（约 30 GB） |

每个新 episode 的 meta 都带 `source`（`exp43:<tag>`）、`source_episode_index`、`seed`、
`global_episode_id`，可用于溯源；`train_lora.py` 的缓存键 `(skill, episode_index)` 在合并数据里
**全局唯一**（新数据 ≥1000，旧数据 <1000）。

## 2. probe（`train_lora.py --mode probe`，`EXP42_DATA=/scratch/lijh/exp43/export`）

```
weights loaded in 60.0s
params total=3403.4M trainable=466.96M (13.72%)
state (4, 32) float64 | actions (4, 16, 32) float64
images base_0_rgb (4,224,224,3), left_wrist_0_rgb (4,224,224,3), right_wrist masked
tokenized_prompt (4,200) mask=162
action range [-0.814, 1.000] | state range [-0.829, 1.018]
action_mask valid per sample: [16, 16, 16, 16]
batch skills: ['B90', 'A90', 'A90', 'A90']
FORWARD LOSS = 2.136661
```

判读：合并后的 314 条 train 数据可以被 OpenPI 接口正确读取并完成一次前向（forward loss 有限、
shape/dtype/掩码/图像三路符合契约），且 batch 里同时抽到 A90 与 B90。

修复点：probe 分支原先直接把 numpy batch 传给模型，触发 `jaxtyping.TypeCheckError`；
已改为 `jax.tree.map(jnp.asarray, obs)` / `jnp.asarray(actions)`。

## 3. 仍未完成

1. **没有更新 normalization 统计**（按约定本轮不改 DROID stats；train-only 统计也未重算）；
2. **没有做 10% 分层的视频内容抽查**（只做了全量"可解码+帧数=行数+亮度/方差"与新旧亮度对比）；
3. **旧 90 条没有做 v1.2 复核**；
4. 未开始任何训练（按要求）。
