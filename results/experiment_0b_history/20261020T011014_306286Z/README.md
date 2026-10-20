# 实验 0B：后继任务几何敏感性检查

本实验让 A 分别产生 z1/z2 两种终态，再执行位置、朝向都不同的 B/C 末端目标。每个条件使用 4 个关节初值做 IK，多起点结果先做限位检查，再做 80 段离散关节插值碰撞采样，路径通过后由原电机控制器执行。保留 direct 与 Reset-to-Canonical 对照。

这只是几何与控制器 sanity check，不是物体交接实验，也不是 VLA 组合泛化证据。路径碰撞检查是离散采样，不构成连续路径安全证明。重复运行确定性场景不等于独立统计样本。

运行：`python scripts/run_experiment_0b.py`。结果在本目录：`multistart_results.csv`（逐起点原始记录）、`summary.csv`、`comparison.png`、`config.json`（参数、种子、探索披露、旧 CSV SHA-256）。不覆盖原实验结果。
