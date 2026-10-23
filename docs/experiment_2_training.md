# 实验 2 · 阶段 3：共享 SmolVLA 微调与原子评测

## 1. 训练配置

| 项 | 值 |
|---|---|
| 数据集 | `data/lerobot_panda_multi`（480 集 / 68 808 帧 / 4 任务） |
| 初始化 | `lerobot/smolvla_base`（VLM 冻结，动作专家可训练） |
| 可训练参数 | 99 880 992（总参 450 046 226） |
| 训练步数 / batch | 20 000 / 8（约 2.33 个 epoch） |
| 学习率 | 1e-4，余弦衰减到 2.7e-5 |
| 保存频率 | 每 2 000 步 |
| 精度/设备 | 单张 A40，bf16（框架默认） |
| 视频解码 | pyav（服务器无系统 ffmpeg，torchcodec 不可用） |
| 耗时 | 12:09 → 14:01，**111 分钟** |
| 产物 | `outputs/smolvla_multi/checkpoints/020000/pretrained_model` |

## 2. 训练曲线

| 步数 | 50 | 2K | 5K | 10K | 15K | 20K |
|---|---:|---:|---:|---:|---:|---:|
| loss | 0.620 | 0.034 | 0.016 | 0.010 | 0.006 | 0.006 |
| grad norm | 5.05 | 0.65 | 0.43 | 0.34 | 0.24 | 0.21 |

loss 在前 2 000 步快速下降，10K 之后基本平台化。训练日志：服务器 `logs/e2_train.log`。

冒烟阶段（20 步）验证了：4 任务 batch 形状正确、forward/backward 正常、checkpoint 可保存并可被 `SmolVLAPolicy.from_pretrained` 重新加载、`select_action` 输出 8 维动作（chunk 50）。

## 3. 原子技能严格评测（20 集/技能）

测试种子 `40000000–40000019`，与采集种子（900000+）完全隔离；严格口径：判定前不替模型张爪、不修正末端、不移动物体。

| 技能 | VLA 成功 | VLA 成功率 | 专家成功 | 专家成功率 |
|---|---:|---:|---:|---:|
| A0 | 20/20 | **100%** | 20/20 | 100% |
| A90 | 20/20 | **100%** | 20/20 | 100% |
| B | 8/20 | **40%** | 18/20 | 90% |
| C | 19/20 | **95%** | 19/20 | 95% |

推理延迟：平均约 11 ms/控制周期（chunk 50，20 Hz 预算 50 ms）。

### 观察

- **A0 / A90 / C 已经追平专家**：同一个共享模型在语言指令条件下能完成三种不同的物理操作，包括 A90 的腕部旋转与 C 的"重新抓取 + 转回长轴 + 入托"。
- **B（推动）是唯一明显落后的技能（40% vs 专家 90%）**。失败以 `push_check` 为主：模型能把物体往 +X 推、常常也能推到墙边，但**推动过程中物体 yaw 变化过大**（>15°），或者最后一段没贴到位（前缘间隙 >2 cm）。
- B 与采集阶段的困难一致：专家在 20 Hz 动作保持下也需要 176 次尝试才凑齐 120 条，说明这个技能对接触时机很敏感，是四个技能里"最不像位置控制"的一个。

## 4. 失败类型（B）

8/20 成功；失败集里：

- 一部分推到墙边但 yaw 超限（物体被推转）；
- 一部分停在半路（前缘间隙 > 2 cm）；
- 没有出现"完全不接触物体"的情况——模型确实在推，只是推不稳。

## 5. 产物

| 位置 | 内容 |
|---|---|
| `outputs/smolvla_multi/checkpoints/020000/pretrained_model` | 最终 checkpoint |
| 服务器 `results/experiment_2/eval_atomic/` | 逐集 JSON + `episodes.csv` |
| 服务器 `results/experiment_2/expert_atomic/` | 同种子专家基线 |
| `docs/experiment_2_data.md` | 数据集说明 |

## 6. 复现

```bash
# 训练（服务器，单卡 A40）
cd ~/simulate_vla
GPU=0 STEPS=20000 BATCH=8 SAVE_FREQ=2000 \
  setsid nohup bash server/train_smolvla_multi.sh > logs/e2_train.log 2>&1 &

# 原子评测
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src venv/bin/python \
  scripts/eval_multi_atomic_vla.py \
  --policy outputs/smolvla_multi/checkpoints/020000/pretrained_model \
  --skills A0 A90 B C --episodes 20 --seed-base 40000000 \
  --out results/experiment_2/eval_atomic

# 专家基线（同种子）
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src venv/bin/python \
  scripts/eval_multi_atomic_vla.py --expert --skills A0 A90 B C --episodes 20 \
  --seed-base 40000000 --out results/experiment_2/expert_atomic
```
