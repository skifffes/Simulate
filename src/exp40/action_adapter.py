"""动作适配器：π₀.₅ DROID 输出 → PyBullet 关节目标 / 夹爪目标。

必须区分的四层（prompt Step 4 明确要求）：
    Policy Action     : 模型返回的 8 维（7 关节增量 + 1 夹爪），已由官方 transform 反归一化
    Adapter Target    : q_target = q_current + Δq，夹爪目标宽度（m）
    Controller Command: simulation.command(q_target) 的 POSITION_CONTROL 目标
    Actual Robot Motion: 物理引擎积分后的真实关节轨迹（由调用方记录）

本文件只负责前两层，并把每一条拒绝理由显式记录下来，绝不用“无限制裁剪”掩盖错误动作。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from . import droid_contract as dc
from .observation_adapter import FINGER_MAX_M

# 官方 DROID 客户端在送入控制器前把动作 clip 到 [-1,1]（examples/droid/main.py:149）。
OFFICIAL_CLIP = 1.0


@dataclass
class ActionAdapterConfig:
    """动作适配的显式参数，全部写进工件（artifact）。"""

    # 单步关节增量上限（rad）：先按官方 clip ±1，再按安全上限收紧。
    official_clip_rad: float = OFFICIAL_CLIP
    max_delta_rad: float = 0.25
    # 夹爪：官方以 0.5 为阈值二值化（examples/droid/main.py:141-146）。
    gripper_threshold: float = 0.5
    gripper_mode: str = 'binary'  # 'binary' | 'continuous'
    finger_max_m: float = FINGER_MAX_M
    # 关节限位安全裕度（rad）
    joint_margin_rad: float = 0.02


@dataclass
class AdaptedAction:
    """一次动作适配的结果（同时是审计记录）。"""

    raw: np.ndarray
    delta_raw: np.ndarray
    delta_applied: np.ndarray
    q_current: np.ndarray
    q_target: np.ndarray
    gripper_value: float
    gripper_target_m: float
    gripper_binary: float
    accepted: bool
    reasons: list = field(default_factory=list)
    clipped_dims: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            'raw': self.raw.tolist(),
            'delta_raw': self.delta_raw.tolist(),
            'delta_applied': self.delta_applied.tolist(),
            'q_current': self.q_current.tolist(),
            'q_target': self.q_target.tolist(),
            'gripper_value': float(self.gripper_value),
            'gripper_target_m': float(self.gripper_target_m),
            'gripper_binary': float(self.gripper_binary),
            'accepted': bool(self.accepted),
            'reasons': list(self.reasons),
            'clipped_dims': list(self.clipped_dims),
        }


def adapt_action(
    action: np.ndarray,
    q_current: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    config: ActionAdapterConfig | None = None,
) -> AdaptedAction:
    """把一条 8 维策略动作适配成关节目标与夹爪目标。

    拒绝规则（返回 accepted=False，调用方必须停止执行该动作并记录）：
      * 维度不是 8
      * 含 NaN/Inf
      * 关节增量超过 official_clip ±1（说明模型输出未按约定反归一化，属于接口错误）
      * 目标关节超出 URDF 限位（经裕度收紧后仍不可达）
    """
    cfg = config or ActionAdapterConfig()
    raw = np.asarray(action, dtype=np.float64).reshape(-1)
    q_current = np.asarray(q_current, dtype=np.float64).reshape(-1)
    lower = np.asarray(lower, dtype=np.float64).reshape(-1)
    upper = np.asarray(upper, dtype=np.float64).reshape(-1)

    if raw.size != dc.ACTION_DIM:
        raise ValueError(f'动作必须是 {dc.ACTION_DIM} 维，收到 {raw.shape}')
    if not np.all(np.isfinite(raw)):
        return AdaptedAction(raw, np.full(7, np.nan), np.zeros(7), q_current, q_current.copy(),
                             float('nan'), float('nan'), float('nan'), False, ['nan_or_inf'])
    if not np.all(np.isfinite(q_current)):
        return AdaptedAction(raw, raw[:7], np.zeros(7), q_current, q_current.copy(),
                             float(raw[7]), float('nan'), float('nan'), False, ['nan_or_inf_state'])

    delta_raw = raw[:7].copy()
    reasons: list[str] = []
    clipped: list[int] = []

    # 1) 官方 clip：超出 ±1 说明接口层没按约定反归一化，拒绝而不是默默裁掉。
    if np.any(np.abs(delta_raw) > cfg.official_clip_rad + 1e-9):
        reasons.append('delta_exceeds_official_clip')
        delta_raw = np.clip(delta_raw, -cfg.official_clip_rad, cfg.official_clip_rad)
        clipped = [int(i) for i in np.where(np.abs(raw[:7]) > cfg.official_clip_rad)[0]]

    # 2) 安全上限：本次实验的实际执行幅度限制。
    delta_safe = np.clip(delta_raw, -cfg.max_delta_rad, cfg.max_delta_rad)
    for i in range(7):
        if abs(delta_safe[i] - delta_raw[i]) > 1e-12:
            if i not in clipped:
                clipped.append(i)

    q_target = q_current + delta_safe

    # 3) 关节限位（带裕度）
    lo = lower + cfg.joint_margin_rad
    hi = upper - cfg.joint_margin_rad
    if np.any(q_target < lo - 1e-9) or np.any(q_target > hi + 1e-9):
        reasons.append('joint_limit_target')
        q_target = np.clip(q_target, lo, hi)
        delta_safe = q_target - q_current

    # 4) 夹爪
    g = float(np.clip(raw[7], dc.GRIPPER_RANGE[0], dc.GRIPPER_RANGE[1]))
    binary = 1.0 if g > cfg.gripper_threshold else 0.0
    if cfg.gripper_mode == 'binary':
        width = binary * cfg.finger_max_m
    else:
        width = g * cfg.finger_max_m

    accepted = not reasons
    return AdaptedAction(raw, raw[:7].copy(), delta_safe, q_current, q_target,
                         g, float(width), binary, accepted, reasons, clipped)


def chunk_to_targets(
    chunk: np.ndarray,
    q_current: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    config: ActionAdapterConfig | None = None,
) -> list[AdaptedAction]:
    """把一个 (horizon, 8) chunk 逐条适配。

    注意：DROID 的动作是相对“当前 state”的增量，chunk 内每一条都是相对上一控制步的增量，
    因此适配时必须逐条递推 q_current，而不是全部相对同一个 q。
    """
    chunk = np.asarray(chunk, dtype=np.float64)
    if chunk.ndim != 2 or chunk.shape[1] != dc.ACTION_DIM:
        raise ValueError(f'chunk 必须是 (T, {dc.ACTION_DIM})，收到 {chunk.shape}')
    out = []
    q = np.asarray(q_current, dtype=np.float64).copy()
    for row in chunk:
        adapted = adapt_action(row, q, lower, upper, config)
        out.append(adapted)
        q = adapted.q_target
    return out


def summarize_chunk(chunk: np.ndarray) -> dict:
    """chunk 的统计摘要（写进工件，便于跨实验比较）。"""
    c = np.asarray(chunk, dtype=np.float64)
    return {
        'shape': list(c.shape),
        'dtype': str(np.asarray(chunk).dtype),
        'finite': bool(np.all(np.isfinite(c))),
        'joint_delta_abs_max': float(np.abs(c[:, :7]).max()) if c.size else 0.0,
        'joint_delta_abs_mean': float(np.abs(c[:, :7]).mean()) if c.size else 0.0,
        'gripper_min': float(c[:, 7].min()) if c.size else 0.0,
        'gripper_max': float(c[:, 7].max()) if c.size else 0.0,
        'gripper_binary_frac': float((c[:, 7] > 0.5).mean()) if c.size else 0.0,
        'per_dim_absmax': np.abs(c).max(axis=0).round(6).tolist() if c.size else [],
    }


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=float)
