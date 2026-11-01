# 实验 4.0：π₀.₅ 基座迁移与 PyBullet 原子技能环境适配

- 最终报告：[results/experiment_4_0/experiment_4_0_report.md](../results/experiment_4_0/experiment_4_0_report.md)
- 环境报告：[results/experiment_4_0/environment_report.md](../results/experiment_4_0/environment_report.md)
- 兼容性对照：[results/experiment_4_0/compatibility_matrix.csv](../results/experiment_4_0/compatibility_matrix.csv)
- 目录说明：[results/experiment_4_0/README.md](../results/experiment_4_0/README.md)

结论：**PASS**（仅指迁移基础设施可用，不代表 π₀.₅ 具备 A90/B90 能力）。

## 一句话结论

官方 `pi05_droid` 已在单张 A40 上真实加载并推理（首次 74 s 含 JIT、稳态 155 ms / 6.45 Hz、
真实显存 8.5 GB）；观测与动作适配器通过 32 项判据；A90/B90 接口检查 8/8 与 9/9；
PyBullet 物理闭环真实跑通并按设计终止（B90 跑满步数预算、A90 被安全层以碰撞终止；物体未被移动）。**未评测任务成功率，未改语言指令，未做训练。**

## 关键接口事实（都带源码证据）

| 项目 | π₀.₅（pi05_droid） | 本仓库（D2） | 适配 |
| --- | --- | --- | --- |
| 图像 | 2 路 224×224×3 uint8 + 第三路占位 mask=False | 2 路 256×256×3 | resize_with_pad + 键名映射 |
| 本体 | 8 维 = 7 关节(rad) + 夹爪[0,1] | 9 维 = 7 关节 + 两指(m) | 夹爪 ÷0.04 |
| 动作 | 15×8 = 7 关节增量(rad/步) + 绝对夹爪[0,1] | 8 维绝对关节角 + 单指宽度 | `q_target=q+Δq`，夹爪 0.5 二值化 |
| 归一化 | quantile（q01/q99 → [-1,1]） | z-score（checkpoint 内） | 全部交给官方 transform |
| 频率 | DROID 15 Hz | 20 Hz | 4.0 保持 20 Hz 并记录差异 |

## 两个环境（不要混）

```bash
# 策略侧：openpi venv（Python 3.11 / jax 0.5.3 / torch 2.6.0），GPU2
bash server/exp40/run_server.sh checkpoint 2 8000

# 仿真侧：simulate_vla venv（Python 3.10 / pybullet / openpi-client），另开一个 shell
/scratch/lijh/simulate_vla/venv/bin/python scripts/exp40/level4_closed_loop.py --scene B90 --ticks 20
```

## 入口脚本

```bash
python scripts/exp40/level1_model_inference.py --mode checkpoint   # 模型独立推理（需 GPU）
python scripts/exp40/level2_offline_sim_obs.py --scene B90         # 真实观测 → 模型（不执行）
python scripts/exp40/level3_adapter_checks.py                     # 适配器判据（含真实 FK）
python scripts/exp40/level4_closed_loop.py --scene B90 --ticks 20  # 有限物理闭环
python scripts/exp40/level5_env_interface.py                      # A90/B90 接口检查
```

## 已知阻塞（4.1 必须处理）

1. 相机视角与 DROID 分布不一致（斜视第三人称 vs 左外侧相机）——最可能的零样本失败原因；
2. 控制频率 20 Hz ≠ DROID 15 Hz；
3. 推理 155 ms/chunk，15 Hz 下不实时；
4. CentOS 7 / glibc 2.17 装不上 lerobot、torchcodec、rerun、torch 2.7.1（本实验用“被调用即报错”的桩绕过，
   仅用于推理；训练必须先解决）；
5. 夹爪 0/1 语义只在仿真侧验证过，需 DROID 侧对照；
6. B90 严格判据（真实 wall contact / 法向力 / 非穿透）尚未实施，见
   [evaluation_gap_analysis.md](../results/experiment_4_0/evaluation_gap_analysis.md)。
