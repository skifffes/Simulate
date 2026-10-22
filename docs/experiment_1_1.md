# 实验 1.1：SmolVLA 严格闭环评测与失败原因诊断

## 1. 目的

实验 1 报告的最高成功率（IID 75%）来自一套未受审计的评测流程，存在两个疑点：

1. 评测器可能在判定成功之前替模型张开夹爪，把"没松手"变成"成功"；
2. 数据采集用 256×256 直接渲染，而评测默认 2 倍超采样再缩小，模型输入分布可能不一致。

本轮不训练、不调参，只做三件事：**把评测改成严格自主、把渲染口径对齐、把失败拆到阶段级**，然后用同一批固定种子重跑，给出可信的成功率与失败原因。

## 2. 结论摘要

| 问题 | 结论 |
|---|---|
| 实验 1 的 75% 是否被辅助松爪污染？ | **没有证据支持污染**。严格模式下 60 集成功率 85%，不低于旧口径；对全部失败集做"补张爪"反事实复跑，**没有一集被救回**。 |
| 完全自主成功率 | **85%（51/60，95% CI 74–92%）**，chunk50；n10 为 78%（47/60，66–87%）。 |
| 主要失败阶段 | **全部集中在"接近-抓取"**：19 次失败里 18 次是"闭爪了但双指没有同时接触工件"，1 次是"全程没下达闭爪指令"；抬升后滑落、放置偏差、释放不稳定在 IID 组一次都没出现。 |
| 采集与评测图像是否一致？ | 采集脚本 `--supersample 1`，旧评测默认 `supersample=2`，**确实不一致**。对齐到 ss=1 后成功率 70% vs ss=2 的 65%（p=1.0，不显著），影响方向与幅度都不足以解释旧结果。 |
| n10 与 chunk50 差异是否稳定？ | 不稳定：60 集上 chunk50 85% vs n10 78%，20 集配对 McNemar p=0.75。**不能据此说哪个更好**。 |
| IID 与真正 OOD 的差距 | IID 85%；受控 OOD 中位置偏移 3 cm 掉到 **10%**，朝向偏移 0.25 rad 掉到 **0%**（专家在同场景 100%），摩擦降到 0.4 仍 **100%**。 |
| 能否进入多技能组合实验？ | **不建议**。原子技能在训练分布内可用，但对位置/朝向的泛化几乎为零；组合任务会把这些误差放大。 |

## 3. 审计与代码改动

### 3.1 严格 / 辅助两套判定（P0）

`scripts/eval_vla_policy.py` 增加 `--mode strict|assisted`：

- **strict**：rollout 结束后先判定，再让物理静置；判定前不调用任何 `set_gripper`、不修正末端、不动工件。额外要求 `autonomous_release` —— 必须出现模型自己下达的张开指令，否则记 `no_autonomous_release`。
- **assisted**：完整保留旧行为（先替模型张开夹爪再判定），只用于解释历史数据。
- **反事实**：strict 失败后，脚本再补一次"替它张爪 + 静置"，记录 `counterfactual_assisted_success`，用来量化"如果旧评测会怎么判"。

旧代码保存在提交 `7ef5f81`，旧结果保存在 `results/experiment_1/`，均未覆盖。

### 3.2 渲染一致性（P0）

`scripts/check_render_consistency.py` 在同一场景、同一初始姿态下分别用 ss=1 与 ss=2 渲染，与数据集视频首帧逐像素比较：

| 对比 | MAE | PSNR |
|---|---:|---:|
| 采集帧 vs ss=1 | 1.79 | 37.7 dB |
| 采集帧 vs ss=2 | 1.94 | 35.2 dB |
| h264 编码噪声底 | 2.13 | — |

采集帧与 ss=1 的差异已经低于视频编码噪声，结论：**采集等价于 ss=1**。可视化见 `results/experiment_1_1/render_check/render_compare.png`。

本轮所有新评测统一 `--supersample 1`；另保留一组 `ss=2` 对照量化差异。

### 3.3 阶段级诊断（P1）

`src/vla/diagnostics.py` 从 PyBullet 真值/接触数据离线提取：接近、首次接触、闭爪、离桌、达到抬升高度、进入目标区、开始张爪、脱离接触、稳定放置，并记录闭爪时的水平/高度偏差、双指接触、滑移、跟踪误差饱和、动作翻转。**这些量只用于分析，不进入策略输入。**

失败主因规则：未闭爪 → `no_gripper_close`；闭爪后无双指接触 → `no_grasp_contact`；抬升 <3 cm → `lift_failed`；滑移 >2 cm → `dropped_in_transport`；未进目标区 → `place_position_error`；未自主张爪 → `no_autonomous_release`；超时 → `timeout`。

### 3.4 受控 OOD 注入

`src/vla/scenario.py` 支持 `shift/yaw/friction` 显式覆盖，一次只改一个因素；目标点、工件尺寸/质量、成功判据全部不变。评测脚本对应 `--ood-shift-x/--ood-shift-y/--ood-yaw/--ood-friction`。

训练分布范围（`make_params`）：位置 ±0.008 m、yaw ±0.10 rad、摩擦 0.8–1.0。OOD 取：位置 +0.03 m（3.75×）、yaw 0.25 rad（2.5×）、摩擦 0.4（低于训练下限一半）。

## 4. 评测协议

| 项 | 值 |
|---|---|
| checkpoint | `outputs/smolvla_panda/checkpoints/015000/pretrained_model`（15 000 步，未重训） |
| 策略控制 | 20 Hz，每个动作保持 12 个物理步（240 Hz） |
| 动作块 | chunk50 = checkpoint 默认 50 步；n10 = 每 10 步重新观测 |
| 渲染 | TINY_RENDERER，256×256，ss=1（与采集一致） |
| 种子 | IID：30000000–30000059；OOD：30000000–30000009（各配置共用） |
| 单集上限 | 260 个控制周期 |
| 专家 | `VLAExpertTask`，同一场景种子、同一相机与判定 |
| 统计 | Wilson 95% 区间；同种子配对用 McNemar 精确检验 |

种子 20261021 起的那批与训练轨迹同种子，只能算拟合检验，本轮**不再**把它当泛化证据；30000000 起这批也**不是 OOD**，只是同分布的 held-out，报告中统一称 IID held-out。

## 5. 结果

### 5.1 严格 vs 旧口径

| 组别 | 集数 | 成功 | 成功率 | 95% CI |
|---|---:|---:|---:|---|
| strict chunk50（ss=1） | 20 | 16 | 80% | 58–92% |
| assisted chunk50（ss=1，旧行为重建） | 20 | 17 | 85% | 64–95% |
| strict n10（ss=1） | 20 | 14 | 70% | 48–85% |
| 实验 1 旧记录 chunk50（ss=2） | 20 | 15 | 75% | 53–89% |
| 实验 1 旧记录 n10（ss=2） | 20 | 10 | 50% | 30–70% |

同种子配对（strict vs assisted，20 对）：都成功 13、仅 strict 3、仅 assisted 4、都失败 0，**McNemar p=1.0**。

更关键的是反事实：对 strict 的全部失败集补做"替它张爪再静置"，**成功数没有任何增加**（IID-20 16→16、n10 14→14、IID-60 51→51）。也就是说，旧评测的自动松爪在这批种子上没有制造过假成功。

### 5.2 渲染口径的影响

同模式、同种子、只改 ss：

| 组别 | 集数 | 成功 | 成功率 |
|---|---:|---:|---:|
| strict n10 ss=1 | 20 | 14 | 70% |
| strict n10 ss=2 | 20 | 13 | 65% |

配对：都成功 8、仅 ss1 6、仅 ss2 5、都失败 1，p=1.0。差异在噪声内，但方向与"ss=1 更贴近训练"一致，因此后续统一用 ss=1。

### 5.3 IID held-out（60 集）

| 组别 | 集数 | 成功 | 成功率 | 95% CI | 自主松爪 | 平均抬升 | 平均放置误差 |
|---|---:|---:|---:|---|---:|---:|---:|
| strict chunk50 | 60 | 51 | **85%** | 74–92% | 60/60 | 0.093 m | 0.031 m |
| strict n10 | 60 | 47 | **78%** | 66–87% | 59/60 | 0.087 m | 0.039 m |
| 专家 | 20 | 20 | 100% | 84–100% | — | 0.100 m | 0.003 m |

成功率条形图（带 95% 区间）：`results/experiment_1_1/success_rate.png`。

### 5.4 受控 OOD

| 配置 | VLA | 专家 | 说明 |
|---|---:|---:|---|
| 位置 +3 cm（x） | 1/10 = **10%** | 10/10 | 场景可完成；模型系统性偏 2 cm |
| yaw 0.25 rad | 0/10 = **0%** | 10/10 | 场景可完成；闭爪位置几乎正中（偏差 1.9 mm），但抓不住 |
| yaw 0.35 rad | 0/10 | 0/10 | 已接近夹爪几何极限，专家也失败，**不作为 OOD 证据** |
| yaw 0.50 rad | 0/10 | 0/10 | 超出几何极限（投影宽度 0.083 m > 开口 0.08 m） |
| 摩擦 0.4 | 10/10 = **100%** | 10/10 | 低摩擦不构成挑战 |

位置偏移下失败集的闭爪水平偏差平均 **20 mm**，而 yaw 0.25 下只有 **1.9 mm**：两者是不同性质的错误——前者是"没走到位"，后者是"走对了但抓不住/不会转腕"。

### 5.5 失败诊断

60 集 IID chunk50 的 9 次失败：`no_grasp_contact` ×9。n10 的 13 次失败：`no_grasp_contact` ×12、`no_gripper_close` ×1。

| 指标 | 失败集 | 成功集 |
|---|---:|---:|
| 闭爪时水平偏差 | 3 mm | 3 mm |
| 平均控制周期 | 170 | 124 |
| 关节跟踪饱和 tick 数 | 16.2 | — |

失败不是"差一点"：失败集平均多跑 46 个周期，且往往在空处闭合后继续尝试。抬升后滑落、放置超差、释放后不稳定在 IID 组为 0 次；旧报告里"78% 失败是没抓起"的结论在严格口径下**依然成立，而且更极端**（100%）。

失败原因分布图：`results/experiment_1_1/failure_primary.png`；逐集数据：`results/experiment_1_1/episodes_all.csv`。

视频（俯视相机，严格模式）：

- 成功：[episode_30000000_ok.mp4](../results/experiment_1_1/strict_chunk50_iid20/episodes/videos/episode_30000000_ok.mp4)
- 成功：[episode_30000012_ok.mp4](../results/experiment_1_1/strict_chunk50_iid20/episodes/videos/episode_30000012_ok.mp4)
- 失败（空抓）：[episode_30000004](../results/experiment_1_1/strict_chunk50_iid20/episodes/videos/episode_30000004_fail_object_not_lifted.mp4)

## 6. 对七个问题的回答

1. **75% 是否受辅助动作影响？** 否。严格判定下 60 集 85%，反事实补张爪没有救回任何失败集。
2. **完全自主成功率是多少？** chunk50 85%（74–92%），n10 78%（66–87%）。
3. **失败主要发生在哪个阶段？** 接近-抓取段；全部是"闭爪但没形成双指接触"，没有一例走到"抬升后滑落"或"释放失败"。
4. **采集与评测图像一致吗？** 采集等价 ss=1；旧评测 ss=2 确实不一致，但对齐后差异不显著（70% vs 65%，p=1.0）。
5. **n10 与 chunk50 差异稳定吗？** 不稳定，p=0.75，60 集上差 7 个百分点。
6. **IID 与真正 OOD 差多少？** IID 85%；位置 +3 cm 10%；yaw 0.25 rad 0%；摩擦 0.4 无影响。
7. **能否进入组合实验？** 不建议。原子技能在分布内可用，但位置/朝向的泛化边界非常窄，组合任务会直接踩在这些边界上。

## 7. 局限

- IID 60 集、OOD 每组 10 集，Wilson 区间仍然很宽（尤其 OOD 的 0/10 只能给出 0–28%）。
- OOD 只做了三个因素的单点，没有做剂量曲线；yaw 的可行上限在 0.25–0.35 rad 之间，未细化。
- "assisted 模式"是按旧代码语义重建的，不能保证与当时的旧脚本逐行一致。
- 失败分类中的 `no_grasp_contact` 依赖闭爪后 20 个 tick 内的双指接触判定，不等于完整的抓持力分析。
- 只在 15 000 步的单个 checkpoint 上评测，没有训练步数消融。

## 8. 复现

```bash
# 服务器：严格自主，IID 60 集（chunk50 / n10）
env GPU=0 MODE=strict SS=1 SEEDS=$(seq -s, 30000000 30000059) OUT=iid60_chunk50 bash server/e11_run.sh
env GPU=3 MODE=strict SS=1 NA=10 SEEDS=$(seq -s, 30000000 30000059) OUT=iid60_n10 bash server/e11_run.sh

# 旧口径对照
env GPU=0 MODE=assisted SS=1 N=20 OUT=assisted20 bash server/e11_run.sh

# 受控 OOD
env GPU=0 MODE=strict SS=1 N=10 SHIFT_X=0.03 OUT=ood_shiftx bash server/e11_run.sh
env GPU=0 MODE=strict SS=1 N=10 YAW=0.25 OUT=ood_yaw025 bash server/e11_run.sh
env GPU=0 MODE=strict SS=1 N=10 FRICTION=0.4 OUT=ood_friction bash server/e11_run.sh

# 本地汇总（读 results/experiment_1_1/*/eval.json）
python scripts/analyze_experiment_1_1.py
```

## 9. 产物清单

| 文件 | 内容 |
|---|---|
| `results/experiment_1_1/episodes_all.csv` | 300 条 episode 的逐集指标 |
| `results/experiment_1_1/summary.csv` / `summary.md` | 15 组成功率 + Wilson 区间 |
| `results/experiment_1_1/paired_tests.csv` | 三组配对 McNemar 检验 |
| `results/experiment_1_1/failures_primary.csv` | 失败主因分布 |
| `results/experiment_1_1/success_rate.png` | 成功率条形图 |
| `results/experiment_1_1/failure_primary.png` | 失败原因分布 |
| `results/experiment_1_1/render_check/` | 渲染一致性度量与对比图 |
| `results/experiment_1_1/*/eval.json` | 每组原始记录（含逐集诊断字段） |
