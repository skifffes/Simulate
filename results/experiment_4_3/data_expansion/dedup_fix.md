# 实验 4.3 去重修复与哈希归档（P0/P1 修复记录）

日期 2026-11-05。回应验收报告的两个代码问题（问题 1 去重、问题 2 哈希覆盖），
并按建议"只读四个无重复目录"重建训练清单。

## 1. 修复方式

新增 `scripts/exp43/build_train_manifest.py`：

- **只读** `data_a90a`、`data_a90b`、`data_b90a`、`data_b90b`；**不读**首批 `data/`
  （那 9 条与 `a90a/b90a` 是同一 (config, seed) 的重采，offset 都是 40）；
- 对 `(skill, seed)` 再做一次集合去重兜底；
- 给每条 episode 分配**全局唯一 `global_episode_id`**（1..253），并保留
  `source_tag / source_episode_index / seed` 作为 provenance，避免不同批次都从 `ep000` 开始造成覆盖；
- 对实际入选的 **253 个 parquet + 506 个 MP4** 逐文件记录 SHA256，
  键是**相对 `/scratch/lijh/exp43` 的完整路径**（不再被同名叶目录覆盖）。

## 2. 结果（去重后，可直接作为训练输入清单）

| 技能 | 新增唯一 episode | 原 train | 合计 train |
| --- | ---: | ---: | ---: |
| A90 | **140** | 30 | **170** |
| B90 | **113** | 31 | **144** |
| 合计 | **253** | 61 | **314** |

`unique_(skill,seed) = 253`，与 episode 数一致（**零重复**）。
失败样本仍是 58 条（A90 8、B90 50），全部留在 `records/`，不进入清单。

**去重后按训练条件**（替换此前未去重的 27/27/17/… 表）：

| A90 目标 | 条数 | B90 配置 | 条数 |
| --- | ---: | --- | ---: |
| A_x-3_y-3 | 26 | low·y015·home | 22 |
| A_x00_y-3 | 26 | low·y015·offset | 29 |
| A_x+3_y-3 | **16** | mid·y021·home | 26 |
| A_x-3_y00 | 24 | mid·y021·offset | 36 |
| A_x00_y00 | 24 | | |
| A_x+3_y00 | 24 | | |

6/6 A90 目标、4/4 B90 训练配置全部覆盖；最弱仍是 `A_x+3_y-3`（16 条）。

## 3. 交付文件

| 文件 | 内容 |
| --- | --- |
| `train_manifest_v2.json` | 253 条唯一 episode：global id、来源目录、源 episode、seed、config、parquet/视频相对路径、metrics |
| `selected_artifacts_sha256.json` | 253 parquet + 506 mp4 的 SHA256（完整路径为键） |

## 4. 仍未完成（明确）

1. **OpenPI 导出未做**：314 条（61 旧 + 253 新）尚未生成 Anchor-16 的 npy/npz 训练缓存，
   也没有验证"全局 id 与训练缓存索引"的对接（reviewer 问题 3 的预防措施只做了一半：
   id 已唯一，但缓存尚未改为按 id/state_path 索引）；
2. **完整 preflight 未做**（维度/有限值/图像帧数/时间顺序/Anchor-16 构造/`train_lora --mode probe`）；
3. **10% 分层内容抽查未做**（0/10 条件抽查）；
4. **原始 90 条 v1.2 复核未做**；
5. “有效动作 ≥10 帧”措辞已确认不准确：代码只检查 `len(att.actions) >= 10`（总帧数），
   本轮报告已改为“动作序列 ≥10 帧”，不再声称“有效运动帧”。
