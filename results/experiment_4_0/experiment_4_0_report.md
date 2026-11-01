# 实验 4.0 报告：π₀.₅ 基座迁移与 PyBullet 原子技能环境适配

> 一句话：**π₀.₅（`pi05_droid`）已在单张 A40 上真实加载并推理，观测/动作适配器与安全层通过判据，
> PyBullet 物理闭环真实跑通并被安全层正常终止；但 π₀.₅ 零样本并未完成 A90/B90，本次也无意评测这一点。**

- 结论等级：**`PASS`**（判据逐条见 §0.1；`PASS` 不等于 π₀.₅ 具备 A90/B90 能力）
- Simulate commit：`e0483131afda5f8ca28f37e1aaaec29adc7fe97f`（实验开始时 HEAD）
- OpenPI commit：`215abfb217dbac7d5f1273282331b9b1866c0479`
- 时间：2026-11-01 19:00–23:00 (UTC+8)，全部数据来自本次真实执行

---

## 0. 交付摘要

| 交付物 | 位置 | 状态 |
| --- | --- | --- |
| 环境报告（含 glibc 2.17 依赖冲突全过程） | `environment_report.md` | ✅ |
| 兼容性对照（11 个组件，逐项带 openpi 源码行号） | `compatibility_matrix.csv` | ✅ |
| checkpoint 清单（20 个 OCDBT 文件，逐个 size+md5） | `checkpoint_manifest.json` | ✅ 0 缺失 |
| Level 1 模型独立推理（GPU 实测） | `model_smoke/` | ✅ |
| Level 2 真实仿真观测→模型 | `observation_checks/` | ✅ |
| Level 3 适配器判据（32 项） | `action_checks/level3_checks.json` | ✅ 32/32 |
| Level 4 有限物理闭环 | `closed_loop_smoke/trace_*.csv` | ✅ 两场景各 13–14 控制步 |
| Level 5 A90/B90 接口 | `closed_loop_smoke/level5_env_interface.json` | ✅ A90 8/8，B90 9/9 |
| B90 评价缺口审计（4.1 用） | `evaluation_gap_analysis.md` | ✅（纯静态） |

### 0.1 PASS 判据逐条核对（prompt §十二）

| # | 判据 | 结果 | 证据 |
| --- | --- | --- | --- |
| 1 | 官方模型在至少一张 A40 上成功加载并推理 | ✅ | `policy ready (checkpoint) in 6.45s`；`jax.devices()=[CudaDevice(id=0)]`（`CUDA_VISIBLE_DEVICES=3`）；`model_smoke/level1_inference.json` |
| 2 | 完整记录 checkpoint 与推理配置 | ✅ | `checkpoint_manifest.json`（20 文件 / 11.576 GiB / md5 全记录）、`level1_inference.json`（config、horizon、dim、jax 版本） |
| 3 | 明确输入维度、状态含义与 normalization | ✅ | 8 维 state = 7 关节(rad) + 归一化夹爪；quantile 归一化；`compatibility_matrix.csv`、`droid_contract.py` |
| 4 | 明确动作表示、单位、坐标系与控制接口 | ✅ | 15×8 = 7 关节增量(rad/步) + 绝对归一化夹爪[0,1]；关节空间，无 IK；`q_target=q+Δq` 送 `POSITION_CONTROL` |
| 5 | Observation Adapter 通过测试 | ✅ | 真实仿真观测：两路 224×224×3 uint8 + 7 + 1，键名与官方客户端一致，同状态 sha256 可复现 |
| 6 | Action Adapter 通过已知动作转换测试 | ✅ | `level3_checks.json` 32/32（含真实 FK、夹爪方向、限位、NaN、单步幅度） |
| 7 | 至少一次安全、有限的 PyBullet 闭环完成 | ✅ | B90 14 步 / A90 13 步，均被安全层正常终止（`unexpected_collision` / `ee_workspace`），轨迹 CSV 完整 |
| 8 | 能保存真实模型输入、动作输出与物理轨迹 | ✅ | `calls_8000/calls/*.npz`（模型实际输入张量 + 动作）、`trace_*.csv`（关节/EE/夹爪/物体逐 tick） |
| 9 | 可复现 | ✅ | 固定种子、固定脚本、`environment_report.md` §6 的一键命令、`logs/` 全量日志 |
| 10 | README 与 manifest 齐全 | ✅ | `README.md`、`artifact_manifest.json`（含 sha256） |

**`PASS` 的边界**：只表示“迁移基础设施具备进入 4.1 的条件”。本实验**没有**计算 A90/B90 成功率，
**没有**改动语言指令，**没有**做任何训练。`closed_loop_smoke` 里物体位置在两次运行后仍为
`[0.55, 0.0, 0.02]`（未被推动）——这正是“模型能跑”与“模型会推”的区别，必须如实记录。

---

## 1. π₀.₅ 是否真正部署成功？

**是（真实部署，不是模拟）**。官方 `openpi` 在本机 A40 上完成了：加载官方 checkpoint → GPU 前向 →
返回 action chunk，且结果无 NaN/Inf、固定噪声下可复现（两次推理逐位相同）。

| 条件 | 结果 | 证据 |
| --- | --- | --- |
| 官方代码可运行 | ✅ | `openpi@215abfb`，`scripts/exp40/serve_policy.py` 用官方 `create_trained_policy` + `WebsocketPolicyServer` |
| checkpoint 可加载 | ✅ | `Finished restoring checkpoint in 5.19 seconds`（服务器日志 `logs/policy_server_checkpoint.txt`）；`load_seconds=6.45`（含 transform 构建） |
| GPU 推理正常 | ✅ | `jax 0.5.3 / jaxlib 0.5.3 / jax-cuda12-plugin 0.5.3`，驱动 550.120、CUDA 12.4 |
| 产生合法 action chunk | ✅ | `(15, 8)` float，finite=True，absmax=0.657（`level1_inference.json`） |
| 可重复 | ✅ | 固定 noise 两次推理 `max_abs_diff = 0.0` |

## 2. 使用哪一个官方 checkpoint？

- **`pi05_droid`**：`gs://openpi-assets/checkpoints/pi05_droid`
- 配置：`src/openpi/training/config.py:631` → `Pi0Config(action_horizon=15, pi05=True)`，`asset_id="droid"`
- 形态：Orbax **OCDBT 分片**，20 个对象，共 **12 429 488 598 B = 11.576 GiB**，逐文件 size 与 md5 见
  `checkpoint_manifest.json`（`n_missing=0`，全部 size 与 GCS 列表一致）
- 本地缓存：`$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_droid/`（`~/.cache/openpi`），
  使 `download.maybe_download()` 直接命中缓存、运行期不再联网
- 另外记录 **`pi05_base`**（`gs://openpi-assets/checkpoints/pi05_base`，约 9.4 GB）作为将来自定义原子技能
  微调的候选基座；**本实验未加载它**
- 下载过程本身是一次工程攻坚：GCS 在该出口极不稳定（单连接 86 B/s ~ 3.7 MB/s 之间跳变、并发会互相压死），
  最终用“单连接 + Range 续传 + 多域名轮询”跑到 **7.6 MB/s**，12.4 GB 用 39 分钟下完（19:54:01 → 20:32:48，
  `logs/download_ckpt6.txt`）

## 3. 使用 JAX 还是 PyTorch，为什么？

**JAX（官方路线）**：

1. 官方 `pi05_droid` 权重就是 JAX/Flax（Orbax OCDBT），直接 `restore_params` + `model.load`，无需权重转换；
2. 后续 4.x 若做 LoRA，官方脚本是 JAX；
3. 本机可行性已验证：`jax 0.5.3` + `jaxlib 0.5.3`（manylinux2014）+ `jax-cuda12-plugin 0.5.3` 在 A40 上正常出设备；
4. 官方钉的 `torch==2.7.1` 只有 manylinux_2_28 wheel，在本机装不上；改用 `torch 2.6.0`（本机实验 1–3 已在用）。
   torch 在推理路径里只服务 `openpi/shared/image_tools.py` 与类型声明，不参与前向。

## 4. 实际占用多少 GPU 显存？

**真实占用约 8.3 GiB**（关闭 XLA 预分配后按进程统计）：

| 测量 | 数值 | 说明 |
| --- | --- | --- |
| 加载完成后（本进程） | **8 456 MiB** | `XLA_PYTHON_CLIENT_PREALLOCATE=false`，`nvidia-smi` 按 pid 过滤 |
| 20 次稳态推理后（本进程） | **8 482 MiB** | 峰值即 8 482 MiB |
| 策略服务器进程（默认 0.6 预分配） | 27 534 MiB | XLA 预留 60% 显存，**不是**模型真实占用 |
| 随机初始化权重（float32） | 27 528 MiB | 仅说明 fp32 随机权重更大，不作为官方口径 |

顺带结论：官方文档“推理 >8 GB”与本机实测一致；**单张 48 GB A40 远远够用**，无需多卡。

## 5. 模型推理速度如何？

| 指标 | 数值 |
| --- | --- |
| 首次推理（含 JAX JIT 编译） | **74.0 s** |
| 稳态 p50 | **155.3 ms** |
| 稳态 mean / p95 / max | 155.0 / 158.6 / 158.7 ms |
| 稳态吞吐 | **6.45 chunk/s**（n=20，warmup=2） |
| websocket 往返（含传输，JIT 后） | 182 ms（Level 2 A90 实测） |
| 服务器内部 `policy_timing.infer_ms` | 74 799 ms（首次，含 JIT） |

对照：DROID 控制频率 15 Hz（66.7 ms/步），本实验仿真 20 Hz（50 ms/步）。
**6.45 chunk/s 意味着不满足 15 Hz 的实时闭环**（每 155 ms 才能出一次 15 步 chunk）。
按 `execute_k=8` 计算，一个 chunk 覆盖 8 个控制步 → 20 Hz 下理论最长可用 400 ms，
因此 155 ms 的推理延迟在 20 Hz 下仍可勉强支撑（但会限制反应速度）。这是 4.1 必须处理的工程约束。

## 6. Simulate 的哪些观测可以直接复用？

| 观测 | 可否复用 | 依据 |
| --- | --- | --- |
| 7 关节角 `sim.q()`（rad） | ✅ 直接复用 | checkpoint 的 state q01/q99（joint4 ∈ [-2.77,-0.45]、joint6 ∈ [1.17,3.47]）与 `panda.urdf` 限位一致，说明同族关节约定 |
| 两指位移（m） | ⚠️ 一次线性映射 | `/0.04` 得归一化夹爪量 |
| `top`/`wrist` RGB | ⚠️ 仅尺寸需适配 | 256×256→224×224（官方 `resize_with_pad`，BILINEAR+零填充），本来就是 RGB |
| 场景随机化 `build_scene` | ✅ 直接复用 | A90/B90 场景与实验 2/3 完全一致 |
| 物体位姿/接触 | ❌ 不是模型输入 | 只用于评价与安全 |

## 7. 哪些观测需要适配？

1. **键名**：`observation.images.top/.wrist` → `observation/exterior_image_1_left` / `observation/wrist_image_left`；
2. **第三路相机槽位**：`right_wrist_0_rgb` 用全零图 + `image_mask=False`（官方 `droid_policy.py:55-61` 同样处理），
   属于官方输入设计，不是“补零凑维度”；
3. **尺寸/值域**：客户端 224×224 uint8；服务端 `ResizeImages(224,224)` → float32 `[-1,1]`；
4. **夹爪标量**：单指位移 ÷ 0.04；
5. **视角**：Simulate 的 `top` 是 45°/30° 斜视第三人称，DROID 是左外侧相机 —— **无法在 4.0 消除**，
   也是下一阶段最可能的失败原因（见 §15）。

## 8. π₀.₅ 的实际输入状态定义是什么？

```
state(8) = [q1 … q7 (rad, Franka Panda 关节序), gripper (0=闭合, 1=全开)]
→ Normalize(quantile): (x - q01)/(q99 - q01)*2 - 1
→ PadStatesAndActions(32): 后 24 维补 0
→ TokenizePrompt(discrete_state_input=True)：pi05 把 state 离散化成 token
```

- 归一化为 **quantile**：`config.py:187` `use_quantile_norm = model_type != PI0`，pi05 为 True；
- checkpoint 自带 state 统计（实测读取）：
  q01 `[-0.828,-0.840,-0.843,-2.773,-1.843,1.172,-2.047,0.000]`，
  q99 `[0.900,1.385,0.692,-0.454,1.732,3.467,2.198,0.991]`；
- 模型实际看到的张量已落盘：`model_smoke/model_inputs.npz` + `calls_8000/calls/*.npz`
  （后者来自策略服务器入口，是真正送进模型的数组，不是另建的一套观测）。

## 9. 模型输出什么类型的动作？

```
actions: (15, 8) float
[Δq1 … Δq7] rad/控制步（相对当前 state 的增量） + [gripper ∈ [0,1]]（绝对值）
```

证据：
- `DroidOutputs` 只保留前 8 维（`droid_policy.py:92`），模型内部宽度 32，8 维之后由 `Unnormalize` 原样透传
  （`transforms.py:175-181`），没有物理含义；
- checkpoint 的 action 统计：前 7 维 q01/q99 关于 0 对称（如 `[-0.458, 0.448]`）、第 8 维 `[0.0, 0.9998]`
  → 前 7 维是增量、第 8 维是绝对归一化量；
- 官方 DROID 客户端：`RobotEnv(action_space="joint_velocity", gripper_action_space="position")`、
  注释 “joint velocity actions (7) + gripper position (1)”、动作 clip 到 [-1,1]（`examples/droid/main.py:80,132,149`）；
  “joint velocity @15 Hz” 等价于“每步关节增量”。
- 实测输出量级合理：Level 1 首条动作 `[0.134,-0.089,-0.125,-0.340,-0.258,0.051,-0.542, 0.259]`，
  B90 真实观测下 `|Δq|max = 0.231`、夹爪 0.79–0.94（倾向张开）——与训练分布一致，没有“未反归一化”的迹象。

## 10. 是否需要 IK 或任务空间控制器？

**不需要**。`pi05_droid` 是关节增量策略：

```
Policy Action (15×8) → Adapter Target q_target = q_now + clip(Δq) → Controller command(q_target) → 实际运动
```

若将来换成 EE 空间的 checkpoint，才需要 IK，并按 prompt §7.2 逐项验证目标位姿含义、姿态误差、可达性、
关节限制、碰撞与非预期构型跳变——4.0 不预先实现这条路径。

## 11. 当前夹爪动作如何处理？

1. 观测：`g = mean(finger1,finger2)/0.04`，夹到 `[1e-4, 0.9999]`；
2. 动作：第 8 维夹到 `[0,1]`，按官方做法以 **0.5 二值化**（`examples/droid/main.py:141-146`）：
   `1 → 单指目标 0.04 m（全开）`，`0 → 0.0 m（闭合）`；同时保留 `continuous` 模式供 4.1 对照；
3. 方向在真实仿真里验证过：`open_m=0.0400`、`closed_m=0.0003`（`action_checks/level3_checks.json`）。

**未验证**：DROID 侧 `gripper_position` 的 0/1 语义是从官方客户端用法推断的，没有 DROID 数据文档逐条核对，
4.1 需要一次显式对照。

## 12. action chunk 如何执行？

| 项目 | 取值 | 依据 |
| --- | --- | --- |
| 一次预测 | 15 步 | `action_horizon=15`（`config.py:631`） |
| 实际执行 | 前 8 步后重推理（`execute_k=8`） | 官方 `open_loop_horizon=8`（`examples/droid/main.py:42`） |
| 缓冲 | `ActionChunker` 保存整个 chunk，游标逐条消费 | `src/exp40/chunk.py` |
| 重置 | `reset()` 清空；回合开始调用 | 同上 |
| 指令切换 | `set_prompt()` 计数 + `clear_on_prompt_change` 开关 | 为 4.2 的 A→B 边界预留，4.0 不实现连接器 |
| 每步物理执行 | 12 个 240 Hz 步 = 50 ms（20 Hz） | `src/vla/config.py` |
| 实测一次运行 | B90 用 2 个 chunk 执行 14 步；A90 用 2 个 chunk 执行 13 步 | `closed_loop_smoke/*.json` |

## 13. 最小闭环是否真正完成？

**是** —— `PyBullet Observation → π₀.₅ → Action Chunk → Action Adapter → Controller → PyBullet`
在真实 checkpoint 下跑通，并且**由安全层正常终止**，不是崩溃、不是超时：

| 场景 | 执行控制步 | 终止原因 | 细节 | 物体最终位置 |
| --- | --- | --- | --- | --- |
| B90 | 14 | `unexpected_collision` | 指尖(link 9)压进地面 `body 0`，穿透 8.18 mm | `[0.55, 0.0, 0.02]`（未移动） |
| A90 | 13 | `ee_workspace` | 末端跑到 `z=0.6208 m`，超出上限 0.60 m | `[0.55, 0.0, 0.02]`（未移动） |

两场景各保存：逐 tick CSV（关节/目标/EE/两指/夹爪/物体位姿/接受标志/停止原因）、
5 个 chunk 的原始数组（`chunks_*.npz`）、安全事件 JSON、调用记录（14 次服务器调用）。

**诚实结论**：π₀.₅ 零样本在这两个场景里既没有抓也没有推，而是把手臂带向桌外/桌下；
安全层按设计挡住了它。这与“相机视角与训练分布不一致 + 训练机器人是真实 Franka/DROID 数据”一致，
但这只是**假设**，4.0 没有做因果验证。

附带发现（已修复并复测）：`sim.collisions()` 只排除父子 link 对，**没排除两指尖互碰(9,10)**，
导致夹爪闭合时被误判为碰撞；`src/exp40/safety.py` 现把 (9,10) 列为预期接触并单独计数（`expected_contacts_seen`）。

## 14. 哪些测试只是静态检查，哪些经过真实仿真执行？

| 项目 | 性质 | 证据 |
| --- | --- | --- |
| 接口契约（键名/维度/归一化/动作语义） | 静态（源码 + checkpoint 统计） | `droid_contract.py`、`compatibility_matrix.csv` |
| 环境安装/依赖冲突定位 | 真实执行（日志） | `environment_report.md`、`logs/install_openpi*.txt` |
| checkpoint 下载与完整性 | 真实执行 | `logs/download_ckpt6.txt`、`checkpoint_manifest.json` |
| Level 1 模型推理 | 真实 GPU 执行 | `model_smoke/level1_inference.json` |
| Level 2 真实观测→模型 | 真实执行（PyBullet + GPU） | `observation_checks/level2_observation_{B90,A90}.json` |
| Level 3 适配器判据 | 真实执行（含真实 FK / 夹爪方向） | `action_checks/level3_checks.json`（32/32） |
| Level 4 闭环 | 真实物理执行 | `closed_loop_smoke/trace_*.csv` |
| Level 5 A90/B90 接口 | 真实执行 | `closed_loop_smoke/level5_env_interface.json`（A90 8/8、B90 9/9） |
| B90 成功判据审计 | **仅静态** | `evaluation_gap_analysis.md` |
| A90/B90 任务成功率 | **未做**（按 prompt 要求不做） | — |

## 15. 下一阶段数据收集与严格评价需要处理什么接口问题？

1. **视觉分布**（第一优先）：DROID = 左外侧相机 + 腕部相机；Simulate `top` = 斜视第三人称。
   4.1 应先量化差异（同状态下图像 embedding 距离 / 直接可视化对照），再决定“加一路仿 DROID 外参相机”
   还是“直接微调”。
2. **控制频率**：DROID 15 Hz vs 仿真 20 Hz；增量动作的时间尺度不同。4.1 应把仿真策略频率改成 15 Hz
   或对 Δq 做时间归一化，否则同一 chunk 的物理含义不同。
3. **夹爪语义**：需要一次显式对照确认 0=闭合、阈值 0.5。
4. **状态零位**：关节限位区间一致只是必要条件，DROID 与 `panda.urdf` 可能存在固定零位偏置；
   4.1 应把同一物理构型两侧的 `q` 直接对比。
5. **推理延迟**：155 ms/chunk 对 15 Hz 不实时；4.1 若要求 15 Hz 闭环，需要 `execute_k` 调大、
   或换更快采样（更少去噪步）、或异步推理。
6. **严格评价**：见 `evaluation_gap_analysis.md`（真实 wall contact、法向力、非穿透、静置窗口、
   路径方向、初始接触排除、阈值校准，共 8 条）。

---

## Observed（实际完成的测试与测量）

1. **环境**：CentOS 7 / glibc 2.17 / 80 核 / 251 GiB / 5×A40 48 GB / 驱动 550.120 / `/scratch` 可用 289 G。
2. **依赖冲突**：官方 `uv sync` 三轮失败——`rerun-sdk 0.23.1`(manylinux_2_31)、`torchcodec 0.4.0`(manylinux_2_28)、
   `torch 2.7.1`(manylinux_2_28)、`numpy>=2`(需 GCC≥9.3，本机 4.8.5)、`dm-tree 0.1.10`(需 CMake)。
   最终用显式依赖集 + `scipy 1.13.1` / `dm-tree 0.1.8` / `torch 2.6.0` / `numpy 1.26.4` 装成。
3. **lerobot 桩**：`openpi.training.checkpoints` 顶层 import `data_loader` → `lerobot`，
   因此提供一个“被调用即报错”的桩包（`server/exp40/lerobot_stub`），推理路径不触碰。
4. **网络**：pypi 镜像 ~3 MB/s；github ~126 KB/s；huggingface.co 不可达；GCS 单连接 86 B/s–7.6 MB/s 剧烈波动；
   并发连接会互相压死。最终下载 12 429 521 366 B，20/20 文件，size 全对，用时 39 分钟。
5. **checkpoint**：`Finished restoring checkpoint in 5.19 s`；`load_seconds=6.45`。
6. **推理**：首次 74.0 s（JIT）、稳态 155 ms（6.45 Hz）、p95 158.6 ms。
7. **显存**：真实占用 8 456 MiB（加载后）/ 8 482 MiB（峰值）；服务器默认预分配 27 534 MiB。
8. **输出**：`(15, 8)`，finite，absmax 0.657（官方 example 观测下）；真实仿真观测下 `|Δq|max 0.131–0.231`，
   夹爪 0.45–0.94。
9. **可重复性**：固定噪声两次推理 `max_abs_diff = 0.0`。
10. **适配器**：32/32 判据通过（含 7 个关节的 FK 响应、夹爪开合方向、限位/NaN/单步幅度拒绝）。
11. **观测**：真实 A90/B90 观测 → 224×224×3 uint8 ×2 + 7 + 1；A90 8/8、B90 9/9 接口检查通过。
12. **闭环**：B90 14 步后 `unexpected_collision`（指尖压桌面 8.18 mm）停止；A90 13 步后 `ee_workspace`
   （z=0.621 m）停止；两次物体都未被移动。
13. **服务器调用**：14 次真实 websocket 推理，输入张量与输出 chunk 全部落盘。

## Supported（源码 + 测试共同支持的判断）

1. `pi05_droid` 的 state = 8 维（7 关节 rad + 归一化夹爪），action = 15×8（7 关节增量 + 绝对归一化夹爪），
   归一化是 quantile，模型内部宽度 32 且仅前 8 维有物理含义（源码 4 处 + checkpoint 统计 + 实测输出量级）。
2. 关节角可直接复用：checkpoint state q01/q99 与 `panda.urdf` 关节限位一致。
3. 夹爪需要 `÷0.04` 归一化与 `0.5` 二值化；方向（1=开、0=闭）在真实仿真中验证过。
4. 观测适配在真实 PyBullet 观测上可用，且同状态可复现（sha256 一致）。
5. 动作适配可判定：`q_target=q+Δq`、限位、单步幅度、NaN/Inf、夹爪方向都有独立判据（32/32）。
6. 安全层能在物理执行前挡住不合约定的动作：随机权重下模型输出 Δq 达 3.5 rad，被
   `delta_exceeds_official_clip` 拒绝并停止（random 模式的对照记录）。
7. 闭环链路（观察→模型→chunk→适配→控制器→物理）真实可用，并能在异常时安全停止。

## Not Verified（没有运行、尚未适配或无法证明的部分）

1. **π₀.₅ 是否具备 A90/B90 能力** —— 未评测；两次闭环里物体都未被移动，但这不构成能力结论。
2. **相机视角/外参**是否与 DROID 训练分布足够接近 —— 未量化。
3. **DROID 夹爪 0/1 语义** —— 仅从官方客户端用法推断。
4. **15 Hz vs 20 Hz** 对行为的影响 —— 未做对照。
5. **B90 严格成功判据** —— 只做静态审计，未实施。
6. **多卡 / LoRA / 全参微调** —— 按 prompt 明确不做。

## Blockers（下一阶段必须解决的工程障碍）

| # | 阻塞项 | 影响 | 建议处理 |
| --- | --- | --- | --- |
| B1 | **视觉视角与 DROID 分布不一致**（斜视第三人称 vs 左外侧相机） | 最可能的零样本失败原因；直接影响 4.1 的数据采集与微调策略 | 4.1 先加一路仿 DROID 外参相机并量化差异，再决定是否微调 |
| B2 | **控制频率 20 Hz ≠ DROID 15 Hz** | 同一 chunk 的物理含义不同，行为速度不可比 | 4.1 把仿真策略频率改为 15 Hz，或对 Δq 做时间归一化 |
| B3 | **推理延迟 155 ms/chunk** | 15 Hz（66.7 ms/步）下不实时 | 增大 `execute_k`、减少去噪步、或异步推理 |
| B4 | **glibc 2.17 锁死依赖**：lerobot / torchcodec / rerun / torch 2.7.1 均无法安装 | 将来若要训练/用 LeRobot 数据集必须换环境 | 用 conda 装 gcc≥9 或换新系统/容器；本实验以桩绕过 |
| B5 | **仿真接触判据不完整**：指尖互碰、指尖压桌面此前未区分 | 会误报/漏报碰撞 | 已在 `safety.py` 加入预期接触白名单并计数；4.1 需要完善接触语义（桌面接触是否合法取决于任务） |
| B6 | **夹爪语义未对照** | 若方向反了，所有抓取数据都白采 | 4.1 用一次显式对照实验确认 |
| B7 | **GCS 出口不稳定**（86 B/s–7.6 MB/s） | 新 checkpoint 下载可能耗时数小时 | 保留 `download_ckpt6.sh`（Range 续传 + 多域名轮询），或预置本地镜像 |

---

## 附：实验 4.1 的需求（只整理，不在本次执行）

1. **多 A 终态**：实验 3.4 只有 1 个唯一 A90 终态（`A_TERMINAL_DIVERSITY_LIMITED`），4.1 必须补足多种真实放置终态。
2. **多 B 目标与启动场景**：沿用 3.1 的 Near/Medium/Far 分层，但要在采集**之前**固定目标集合与种子预算。
3. **数据采集与质量控制**：原子技能数据需同时记录 `q / EE 6D 位姿 / 夹爪 / 完整场景状态`——
   4.0 的 `level4` trace CSV 已经具备这四类字段，可直接复用为采集模板。
4. **严格物理完成判据**：按 `evaluation_gap_analysis.md` §3 的 8 条改造，并用专家成功/失败各 ≥20 例校准后冻结阈值。
5. **组合划分独立性**：训练/验证/测试的技能组合划分必须在采集前固定。
6. **微调可行性**：`pi05_base` 作为基座（路径已记录），LoRA 需 >22.5 GB 显存，单张 A40 48 GB 足够；
   但受 B4 限制，训练环境需要先解决 lerobot 依赖。
