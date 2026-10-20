# 实验 0C 失败样例（摩擦不足）

这是负对照里的一格，用来肉眼确认“抓取靠的是真实摩擦”，不是隐藏的固定约束。

- 参数：工件 `lateralFriction = 0.2`，每指夹持力上限 `2 N`，其余参数与默认场景完全相同。
- 摩擦上限 `2μF = 2 × 0.2 × 2 = 0.8 N`，小于工件重力 `0.12 × 9.81 = 1.18 N`，理论上前者撑不住后者。
- 结果：`success = False`，`failure = object_not_lifted`，`lift_height_m ≈ -2e-6`，`slip_m = 0.100 m`。

## 画面里发生了什么

`keyframes/` 与 `keyframes.png` 依次是张开 → 高位接近 → 预闭合 → 下降到抓取位 → 闭合 → 抬升 → 失败瞬间。

`07_failure.png` 是失败瞬间：末端已经升到 `z = 0.155 m`（比抓取位高 10 cm），工件仍停在桌面 `z = 0.04 m`。两指确实压住了工件（接触力 2.00 / 2.00 N，等于电机力上限），但在手指上升时工件沿指面滑脱，接触在 `t ≈ 4.03 s` 归零。

`trajectory.csv` 里可以看到三条同时发生的证据：

- `ee_z` 从 0.055 稳定升到 0.155；
- `obj_z` 全程 0.040，没有离开桌面；
- `contact_force_1/2` 在 2.0 N 保持约 0.13 s 后掉到 0。

`slip_m = 0.1` 就是“夹爪上升 10 cm、工件原地不动”的相对位移，因此这次同时满足 `object_not_lifted` 和 `object_slipped`，判定按顺序报了前者。`place_*` 为 `nan`/`null`，因为流程在抬升阶段就终止，没有走到放置。

## 复现

```powershell
Set-Location 'D:\python\simulate'

# 实时窗口亲眼看
& '.\.conda312\python.exe' '.\scripts\watch_experiment_0c.py' --friction 0.2 --force 2 --speed 1

# 重新生成这里的离线数据与 GIF
& '.\.conda312\python.exe' '.\scripts\run_experiment_0c.py' --episodes 1 --friction 0.2 --force 2 --out results/experiment_0c_failure
```

窗口左上角 HUD 会实时显示 `摩擦上限 2μF` 与 `重力`，失败后停在同一画面并给出 `失败原因`。成功场景（默认 `μ=1.2`、`20 N`）的命令见 `results/experiment_0c/README.md`。
