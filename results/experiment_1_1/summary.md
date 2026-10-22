# 实验 1.1 汇总（自动生成）

| 组别 | 集数 | 成功 | 成功率 | 95% Wilson | 自主松爪 | 平均抬升/m | 平均放置误差/m |
|---|---:|---:|---:|---|---:|---:|---:|
| VLA strict · chunk50 · IID-20 | 20 | 16 | 80% | 58–92% | 20/20 | 0.087 | 0.0403 |
| VLA strict · n10 · IID-20 | 20 | 14 | 70% | 48–85% | 19/20 | 0.085 | 0.0427 |
| VLA assisted · chunk50 · IID-20 | 20 | 17 | 85% | 64–95% | 20/20 | 0.101 | 0.0185 |
| VLA strict · n10 · ss2 · IID-20 | 20 | 13 | 65% | 43–82% | 17/20 | 0.069 | 0.0633 |
| VLA strict · chunk50 · IID-60 | 60 | 51 | 85% | 74–92% | 60/60 | 0.093 | 0.0309 |
| VLA strict · n10 · IID-60 | 60 | 47 | 78% | 66–87% | 59/60 | 0.087 | 0.0394 |
| Expert · IID-20 | 20 | 20 | 100% | 84–100% | — | 0.100 | 0.003 |
| VLA strict · OOD shift-x 3cm | 10 | 1 | 10% | 2–40% | 10/10 | 0.074 | 0.0795 |
| VLA strict · OOD yaw 0.25rad | 10 | 0 | 0% | 0–28% | 10/10 | 0.017 | 0.185 |
| VLA strict · OOD yaw 0.5rad (超几何极限) | 10 | 0 | 0% | 0–28% | 10/10 | 0.009 | 0.1975 |
| VLA strict · OOD friction 0.4 | 10 | 10 | 100% | 72–100% | 10/10 | 0.106 | 0.0026 |
| Expert · OOD shift-x 3cm | 10 | 10 | 100% | 72–100% | — | 0.100 | 0.0024 |
| Expert · OOD yaw 0.25rad | 10 | 10 | 100% | 72–100% | — | 0.089 | 0.0054 |
| Expert · OOD yaw 0.5rad (超几何极限) | 10 | 0 | 0% | 0–28% | — | 0.000 | - |
| Expert · OOD friction 0.4 | 10 | 10 | 100% | 72–100% | — | 0.100 | 0.0027 |

## 同种子配对检验（McNemar 精确检验，双尾）

| 比较 | 配对数 | 都成功 | 仅前者 | 仅后者 | 都失败 | p |
|---|---:|---:|---:|---:|---:|---:|
| strict chunk50 vs strict n10 (IID-20) | 20 | 10 | 6 | 4 | 0 | 0.7539 |
| strict vs assisted chunk50 (IID-20) | 20 | 13 | 3 | 4 | 0 | 1.0 |
| ss1 vs ss2 n10 (IID-20) | 20 | 8 | 6 | 5 | 1 | 1.0 |

## 主要失败原因

- **VLA strict · chunk50 · IID-20**：no_grasp_contact×4
- **VLA strict · n10 · IID-20**：no_grasp_contact×4, dropped_in_transport×2
- **VLA assisted · chunk50 · IID-20**：dropped_in_transport×2, no_grasp_contact×1
- **VLA strict · n10 · ss2 · IID-20**：no_grasp_contact×6, dropped_in_transport×1
- **VLA strict · chunk50 · IID-60**：no_grasp_contact×8, dropped_in_transport×1
- **VLA strict · n10 · IID-60**：no_grasp_contact×11, dropped_in_transport×2
- **VLA strict · OOD shift-x 3cm**：place_position_error×6, no_grasp_contact×3
- **VLA strict · OOD yaw 0.25rad**：no_grasp_contact×9, dropped_in_transport×1
- **VLA strict · OOD yaw 0.5rad (超几何极限)**：no_grasp_contact×10
- **Expert · OOD yaw 0.5rad (超几何极限)**：tracking_error×10
