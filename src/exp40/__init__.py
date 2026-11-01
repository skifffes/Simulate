"""实验 4.0：Simulate × π₀.₅ 迁移层。

模块划分（全部与实验 1–3 的代码解耦，只读复用 `simulation` / `vla` 的仿真接口）：

- `droid_contract`       π₀.₅ DROID checkpoint 的真实输入/输出约定（源码与 norm_stats 证据）
- `observation_adapter`  PyBullet 观测 → DROID 策略输入
- `action_adapter`       策略动作 → 关节目标 / 夹爪目标（含拒绝规则）
- `safety`               关节限位、工作空间、单步幅度、NaN/Inf、碰撞、超时
- `chunk`                action chunk 缓冲与重推理时机
- `policy_client`        openpi websocket 客户端封装 + 输入输出审计
"""
from __future__ import annotations

__all__ = [
    'action_adapter',
    'chunk',
    'droid_contract',
    'observation_adapter',
    'policy_client',
    'safety',
]
