# 实验 4.0 环境报告（A40 服务器 · π₀.₅ 部署侧）

本文件只记录**实际执行过**的命令与**实测**数据。凡是推断出来的内容都单独标注“推断”。

- 采集时间：2026-11-01 19:00–23:00（UTC+8）
- 服务器：`222.29.51.96`（SSH 密钥 `~/.ssh/simulate_a40`；凭据只从环境变量读取，不落仓库）
- Simulate commit：`e0483131afda5f8ca28f37e1aaaec29adc7fe97f`（实验开始时 HEAD）
- OpenPI commit：`215abfb217dbac7d5f1273282331b9b1866c0479`
- OpenPI 源码路径：`/scratch/lijh/exp40/openpi`

---

## 1. 系统与硬件（实测）

| 项目 | 实测值 | 命令 |
| --- | --- | --- |
| 主机名 | `localhost.localdomain` | `hostname` |
| 内核 | `Linux 3.10.0-1160.el7.x86_64` | `uname -a` |
| 发行版 | CentOS Linux 7 (Core) | `cat /etc/os-release` |
| glibc | **2.17** | `ldd --version` |
| CPU | 80 核 | `nproc` |
| 内存 | 251 GiB（启动时已用 8 GiB） | `free -g` |
| 根分区 | 382 G，可用 357 G | `df -h /` |
| /scratch | 15 T，**可用 289 G（已用 98%）** | `df -h /scratch` |
| GPU | 5 × NVIDIA A40 46068 MiB，全部空闲 | `nvidia-smi` |
| 驱动 / CUDA | 550.120 / 12.4 | `nvidia-smi` |
| 拓扑 | GPU0-2 同 NUMA0，GPU3-4 同 NUMA1；无 NVLink | `nvidia-smi topo -m` |

**glibc 2.17 是本实验最大的环境约束**：多个现代 wheel 只提供 `manylinux_2_28` / `manylinux_2_31`，
在 CentOS 7 上无法安装（详见第 4 节）。

## 2. Python / 运行时

服务器上原本没有任何可用的 openpi 环境（`/scratch/lijh/openpi` 不存在），但已有 17 个 conda 环境
（`/scratch/lijh/.conda/envs/*`，3.9–3.13）与实验 1–3 使用的 `simulate_vla/venv`（Python 3.10.18，
torch 2.6.0+cu124，lerobot 0.3.3，pybullet 202010061）。

按 prompt §2.3 的要求，本实验采用**两个独立运行时**，通过官方 Policy Server / Client 通信：

| 运行时 | 位置 | 关键版本 | 职责 |
| --- | --- | --- | --- |
| Policy Runtime | `/scratch/lijh/exp40/openpi/.venv`（Python 3.11.16，由 uv 安装的 python-build-standalone） | jax 0.5.3 / jaxlib 0.5.3 / flax 0.10.2 / numpy 1.26.4 / scipy 1.13.1 / torch 2.6.0+cu124（仅图像工具用） | 加载 π₀.₅，GPU 推理，官方 websocket server |
| Simulator Runtime | `/scratch/lijh/simulate_vla/venv`（Python 3.10.18） | pybullet 202010061 / numpy 2.2.6 / torch 2.6.0+cu124 / lerobot 0.3.3 / msgpack 1.2.4 / websockets 16.1.1 / openpi-client 0.1.0 | PyBullet 场景、观测、控制器、官方 websocket client |

**为什么不进程内推理**：openpi 需要 jax 0.5.3 + numpy<2，而现有仿真环境是 numpy 2.2.6 + torch 2.6 + lerobot 0.3.3。
把两者装进同一个 venv 会同时触碰 numpy 主版本冲突与 lerobot 的 torchcodec/rerun 依赖，
而这两个依赖在 glibc 2.17 上装不上（第 4 节）。分离后两边都不需要改动，且与官方 `docs/remote_inference.md`
的推荐架构一致。

## 3. 网络实测（这是本实验真正的瓶颈）

| 目标 | 结果 | 备注 |
| --- | --- | --- |
| `pypi.tuna.tsinghua.edu.cn` | 200，约 **3 MB/s** | 主索引，`uv` 全程使用 |
| `github.com` | 200，约 **126 KB/s** | 足以 clone openpi（浅克隆） |
| `huggingface.co` | **不可达（000）** | 被墙 |
| `hf-mirror.com` | 200 | 单连接约 0.17 MB/s，4 并发可到 6.7 MB/s |
| `storage.googleapis.com`（官方 checkpoint 所在） | 可达但**极不稳定** | 见下 |
| 代理 `127.0.0.1:7897`（prompt 提到的本地代理） | 端口开放，但本机 curl 经它仍 000 | 未使用 |

GCS 的实测分布（同一对象、同一网络、不同时刻）：

```
4.4 KB 文件，无 Range：2.4 s / 23.5 s / 52 s / 超时 200 s
16 MB range，storage-download + IPv4：1.5 MB/s / 3.7 MB/s / 0 B（超时）
12 路并发分段：合计 28 KB/s（互相压死）
```

结论与做法：
1. **不带 Range 的普通 GET 经常 0 字节超时**，带 `Range` 的请求成功率显著更高；
2. **并发连接会互相压死**（该出口对同一目标的并发极不友好），因此改为单连接；
3. 单连接 + Range 续传 + 多域名轮询时，实测跑出 **7.59 MB/s**
   （`params/ocdbt.process_0/d/213a56...` 2.24 GB / 295 s），这是最终采用的方案：
   `server/exp40/download_ckpt6.sh`。

## 4. 依赖安装：与官方 `uv sync` 的偏离（逐条有日志证据）

官方流程 `GIT_LFS_SKIP_SMUDGE=1 uv sync` 在本机**失败**，共三轮，原因全部是 glibc 2.17：

| 轮次 | 失败包 | 官方 wheel 平台 | 处理 |
| --- | --- | --- | --- |
| 1 | `rerun-sdk==0.23.1`（lerobot 的可视化依赖） | 仅 `manylinux_2_31` | `--no-install-package rerun-sdk` |
| 2 | `torchcodec==0.4.0`（lerobot 的视频解码依赖） | 仅 `manylinux_2_28` | 放弃 `uv sync` 路线 |
| 3 | `torch==2.7.1` | 仅 `manylinux_2_28`；而 `torch==2.6.0` 是 `manylinux1`（=2.5） | 改钉 2.6.0（本机实验 1–3 已在用） |
| 3 | `numpy>=2`（被 scipy 1.17 的源码构建拉起，需要 GCC≥9.3，本机 GCC 4.8.5） | — | 钉 `scipy==1.13.1` + `numpy==1.26.4` |
| 3 | `dm-tree==0.1.10`（无 cp311 wheel，需 CMake，本机无 cmake） | — | 钉 `dm-tree==0.1.8`（有 manylinux2014 wheel） |

最终采用的安装方式（`server/exp40/install_openpi{3,4,5}.sh`）：

```bash
uv venv --python 3.11 --seed .venv
uv pip install --python .venv/bin/python numpy==1.26.4 torch==2.6.0
uv pip install --python .venv/bin/python "jax[cuda12]==0.5.3" scipy==1.13.1 dm-tree==0.1.8
uv pip install --python .venv/bin/python <openpi 运行时依赖，逐包安装，--only-binary=:all:>
uv pip install --python .venv/bin/python -e packages/openpi-client --no-deps
uv pip install --python .venv/bin/python -e . --no-deps
```

**未安装**（都属于训练 / 数据加载 / 可视化路径，推理不需要；已核对 import 图）：
`lerobot`、`wandb`、`gym-aloha`、`torchcodec`、`rerun-sdk`、`tensorflow`、`dlimp`。

### 4.1 一个必须显式记录的桩（stub）

`openpi/training/checkpoints.py:16` 在模块顶层 import `openpi.training.data_loader`，
而后者在模块顶层 `import lerobot.common.datasets.lerobot_dataset`。也就是说**即使只做推理**，
`policy_config.create_trained_policy` 也会强制 import lerobot。

因此本实验提供一个**只到“可 import”为止**的桩包：`server/exp40/lerobot_stub/lerobot/...`。
其中 `LeRobotDataset` / `LeRobotDatasetMetadata` 一旦被真正调用会立刻抛 `RuntimeError`
（不会静默返回空数据）。推理路径不会触碰这两个符号。
桩通过 `PYTHONPATH=/scratch/lijh/exp40/lerobot_stub` 注入，**不写进 venv**，以便一眼可见。

## 5. GPU 使用（符合 prompt §2.2 原则）

- 只用**一张** A40：`CUDA_VISIBLE_DEVICES=2`（GPU2 当时 4 MiB 占用，最干净）；
- 不干扰其他用户：实验全程其余 4 张卡保持 1 MiB；
- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.6`（默认预分配 90% 会白占 41 GB）；
- 不做多卡、不做 LoRA、不做全参微调。

实测显存（`nvidia-smi --query-compute-apps=used_memory`）：

| 阶段 | 显存 |
| --- | --- |
| 随机初始化权重（float32）加载完成 | 27 528 MiB |
| 首次推理（含 JIT 编译）后 | 27 528 MiB |
| 20 次稳态推理后 | 27 536 MiB |
| 官方 bf16 checkpoint（见 experiment_4_0_report.md） | 见报告 |

## 6. 复现清单

```bash
# 0) 环境（一次性）
bash server/exp40/install_openpi3.sh    # venv + jax + torch
bash server/exp40/install_openpi4.sh    # 钉版本修依赖
bash server/exp40/install_openpi5.sh    # 逐包安装 + 自检
bash server/exp40/install_client.sh     # 仿真 venv 装 openpi-client

# 1) 官方 checkpoint（约 12.4 GB，单连接 Range 续传）
bash server/exp40/download_ckpt6.sh

# 2) 起策略服务器（GPU2，端口 8000）
bash server/exp40/run_server.sh checkpoint 2 8000

# 3) 分级验证
/scratch/lijh/exp40/openpi/.venv/bin/python scripts/exp40/level1_model_inference.py --mode checkpoint
/scratch/lijh/simulate_vla/venv/bin/python scripts/exp40/level2_offline_sim_obs.py --scene B90
/scratch/lijh/simulate_vla/venv/bin/python scripts/exp40/level3_adapter_checks.py
/scratch/lijh/simulate_vla/venv/bin/python scripts/exp40/level4_closed_loop.py --scene B90 --ticks 20
/scratch/lijh/simulate_vla/venv/bin/python scripts/exp40/level5_env_interface.py
```

## 7. 偏离与风险（明确记录，不含糊）

1. 与官方 `uv sync` 的依赖集不同（第 4 节）：少 lerobot/wandb 等非推理依赖，多一个 lerobot 桩。
   风险：若将来要用 `compute_norm_stats.py` 或训练，必须重新解决 lerobot 的 glibc 依赖。
2. `torch 2.6.0` 而非官方钉的 `2.7.1`。推理路径的 torch 只用于 `openpi/shared/image_tools.py`
   与类型声明，不影响 JAX 前向；但这是**与官方 lock 不一致**，已记录。
3. 控制频率 20 Hz（仿真）vs DROID 15 Hz（训练）：4.0 不修正，只记录。
4. 相机视角与 DROID 不同（第三人称 45°/30° vs DROID 左外侧相机），视觉分布差异无法在 4.0 消除。
