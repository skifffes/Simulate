# Panda 技能交接最小实验

已在本机 Windows 11 原生 GUI 和 DIRECT 模式实际运行。无需 ROS、WSL、模型权重或真机。

## 环境

原系统 Python 3.13 无合适 PyBullet wheel；尝试 Python 3.12 后发现 PyPI 的 PyBullet 3.2.7 同样没有 Windows wheel。最终使用项目独立 Conda 环境 `.conda312`，不修改 base：Python 3.12.15、PyBullet 3.2.5、NumPy 2.5.4、Pillow 12.3.0。实际包 URL 锁定在 `results/conda-explicit-win64.txt`。

硬件：16 GB 内存、RTX 5060 Laptop 约 8 GB 显存。GUI 实测由 Intel 集显 OpenGL 渲染；截图使用 TinyRenderer，不要求 CUDA。

## 运行（PowerShell，不是 Linux Shell）

在 `D:\python\simulate` 打开 PowerShell：

```powershell
Set-Location 'D:\python\simulate'
& '.\.conda312\python.exe' '.\scripts\run_demo.py'
& '.\.conda312\python.exe' '.\scripts\run_experiment.py' --trials 10 --seed 20261019
& '.\.conda312\python.exe' '.\scripts\plot_results.py'
```

演示窗口依次播放 `z1→B`、`z1→C`、`z2→B`、`z2→C` 四种直接交接，画面标注当前阶段和成功/失败原因，不写数据或截图；加 `--reset` 则每组先从 A 终态运动回 HOME 再执行后继。`Simulation.reset()` 支持在现有连接中重新初始化。实验覆盖 8 个条件（4 组直接交接及各自恢复对照），每组 10 次，总计 80 次。增加 `--trials` 可增加次数。程序覆盖同名结果；需要保留旧实验时请先复制 `results`。

如需重建环境（以下仍为 PowerShell）：

```powershell
& 'D:\miniconda3\Scripts\conda.exe' create --prefix 'D:\python\simulate\.conda312' --file 'D:\python\simulate\results\conda-explicit-win64.txt' --yes
```

仅在环境尚不存在时重建。`requirements.txt` 记录版本，但本机不要直接用 pip 安装 PyBullet；使用已验证的 Conda 包。

## 任务与成功条件

立方体固定在 `(0.55, 0, 0.025)`，边长 0.05 m。A 的公共位置目标为 `(0.43, 0, 0.34)`；z1/z2 分别使用绕竖轴不同方向的末端姿态。B 接近立方体上方 `(0.55, 0, 0.36)`；C 接近前方 `(0.55, -0.22, 0.12)`。目标均在 `src/skills.py` 中。

A 的公共任务条件是接近同一区域且无非预期碰撞；两种实现还分别要求各自的姿态追踪成功。运动通过 IK、关节插值和电机位置控制完成，只有每次试验的共同初始化使用 `resetJointState`。恢复基线通过电机移动到公共 HOME 关节角，而非瞬移。

关节追踪误差范数 < 0.06 rad、末端位置误差 < 0.025 m、姿态角误差 < 0.15 rad，且没有检测到越界或非预期碰撞，才算末端动作成功。固定时间步 1/240 s；插值后统一等待 60 步收敛。碰撞检测排除底座支撑和已知相邻连接几何重叠，并不是完备的连续碰撞检测。

## 实测结果

|条件|整链成功|平均交接时间（仿真秒，含恢复）|关节行程（rad）|
|---|---:|---:|---:|
|直接 z1→B|10/10|0.846|0.571|
|直接 z1→C|10/10|1.579|1.276|
|直接 z2→B|10/10|3.629|3.241|
|直接 z2→C|10/10|3.908|3.511|
|恢复 z1→B|10/10|3.958|3.316|
|恢复 z1→C|10/10|4.763|4.088|
|恢复 z2→B|10/10|3.967|3.326|
|恢复 z2→C|10/10|4.771|4.098|

A 在所有条件均成功，8 个条件全部 10/10。z2 直接交接时 PyBullet 的局部 IK 返回第 7 关节 5.483 rad，超出 ±2.967 rad 限位；但该值与 -0.801 rad 相差 2π，属于同一物理构型。控制器现在做 2π 等价折算后再判断限位，`results.csv` 的 `ik_wrapped` 记录该情况：只有 z2 直接交接两组为真，恢复后两组不需要折算。

**此前的“z2 直接交接 0/10 失败”结论是求解器角度未归一的假阴性，已作废。** 修正后四组直接交接均能到达 B/C，z2 的差异只体现为更大的关节行程和更长的仿真耗时（约多 2.3–3.1 秒、多 2.2–2.7 rad），不是可行性差异。z1 在 B、C 上都比 z2 更省，因此**没有出现“B 偏好 z1、C 偏好 z2”的交叉现象**；当前场景中 z1/z2 的差别只是腕部绕转量，与后继目标的几何方向没有耦合。

这些是确定性场景的重复运行，不是 10 次独立随机物理试验，不应解释为鲁棒成功概率；10 次重复结果完全一致，表中数值是确定性复现值而非分布估计。轨迹长度为逐时间步实际关节变化的欧氏范数之和，单位 rad，不是末端空间路程。`joint_tracking_error_rad` 记录关节追踪误差；`position_error_actual_m` 记录末端位置误差。

## 输出

- `results/demo.png`、`demo_log.json`：里程碑 1 的旧演示记录；当前 `run_demo.py` 不写文件。
- `results/results.csv`：逐次记录，包括 A 终态、IK 是否经 2π 折算、失败原因和恢复成本。
- `results/summary.csv`：8 组汇总，包含整链成功、`ik_wrapped_rate` 和恢复成本。
- `results/experiment_config.json`：种子、次数及时间步。
- `results/comparison.png`：由真实汇总绘制的成功率与时间对比。
- `docs/experiment_notes.md`：指标边界和学习要点。
- `docs/vla_next_step.md`：官方资料核对与下一阶段建议。

没有录制视频。`src/policy.py` 预留 `Policy.act(observation)` 抽象接口，但当前实验直接使用脚本控制器，尚未接入 VLA。

**当前仅为技能交接的几何/运动学代理测试，不涉及物体操作，更不是 VLA 未见技能组合泛化的证据。**
