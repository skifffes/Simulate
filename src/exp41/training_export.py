"""实验 4.1 训练导出（Data Preflight）：9 维原始 state/action → π₀.₅ 的 8 维 state + 8 维训练动作。

**关键修正（review #1）**：训练用的夹爪动作必须来自**专家下发的动作目标** `action[:, 7]`，
不能来自实际观测 `state[:, 7:9]`。后者是"现在张开多少"，不是"要它张开多少"；
抓取期间两者差异极大（观测 1.0 张开 vs 目标 0.425 闭合）。

契约（`exp4.1-training-export-v1`）：
    state8[0:7]  = q1..q7 (rad)                      —— 与 4.0 的 DROID 状态前 7 维一致
    state8[7]    = clip(mean(state[7:9]) / 0.04, 0, 1)   —— 观测夹爪（归一化，0=闭合 1=全开）
    action8[0:7] = action[0:7] - state[0:7]           —— 关节增量（rad / 控制步）
    action8[7]   = clip(action[7] / 0.04, 0, 1)       —— **动作目标**夹爪（归一化）
    action8[7]_binary = (action8[7] > 0.5)            —— 可选二值化（官方 client 的下发方式）

`state8` 与 `action8` 都保留连续版本；二值化只用于"与预训练策略下发方式对齐"的对照，
不要同时混用连续与二值标签。

**两种动作表示（必须区分，独立审计 P1）**：
  * `single_step`（本模块 `build_training_arrays` 的输出，也是 DROID / π₀.₅ 预训练的动作语义）：
    第 t 个样本的动作 = `A[t] - S[t]`，即"相对**当前**状态的单控制步增量"。
  * `anchor_horizon`（实验 4.2 实际训练用的标签，见 `train_lora.make_item`）：
    第 k 行 = `A[t+k] - S[t]`，即"相对**窗口锚点** q_t 的跨 k+1 步累计偏移"。
  两者只在 k=0 相等；4.2 部署端用 `chunk_to_targets(anchor_reference='fixed')` 与后者对齐，
  因此 `single_step` 的归一化跨度统计**不能**直接当作 4.2 训练标签的跨度（审计更正：
  单步 Δq 跨度是 DROID 的 2.4–6.1%，而真实 16 步锚点标签是 17–76%）。
  需要后者时用 `build_anchor_horizon_arrays`。
"""
from __future__ import annotations

import numpy as np

EXPORT_VERSION = 'exp4.1-training-export-v1'
FINGER_MAX_M = 0.04          # 单指 0..0.04 m（与 exp40.observation_adapter.FINGER_MAX_M 一致）
GRIPPER_BINARY_THRESHOLD = 0.5


def gripper_from_state(state: np.ndarray) -> np.ndarray:
    """观测夹爪（归一化）——只用于 state，不作为训练动作。"""
    s = np.asarray(state, dtype=np.float64)
    return np.clip(s[..., 7:9].mean(axis=-1) / FINGER_MAX_M, 0.0, 1.0)


def gripper_from_action(action: np.ndarray) -> np.ndarray:
    """**训练动作**夹爪（归一化）——来自专家下发的单指宽度目标。"""
    a = np.asarray(action, dtype=np.float64)
    return np.clip(a[..., 7] / FINGER_MAX_M, 0.0, 1.0)


def build_anchor_horizon_arrays(states: np.ndarray, actions: np.ndarray,
                                horizon: int = 16, stride: int = 4) -> dict:
    """实验 4.2 实际训练用的标签：`anchored[t,k,:7] = A[t+k,:7] - S[t,:7]`。

    与 `train_lora.make_item` 完全一致：**夹爪维度也逐行取未来目标**
    `g_action[t+k]`（审计 P1：早先写成常数 `g_action[t]`，会破坏"接近→闭合→张开"窗口的未来标签）。
    这是"固定锚点"语义；部署端必须用 `chunk_to_targets(anchor_reference='fixed')`。
    """
    S = np.asarray(states, dtype=np.float64)
    A = np.asarray(actions, dtype=np.float64)
    if S.ndim != 2 or S.shape[1] != 9:
        raise ValueError(f'state 必须是 (T,9)，得到 {S.shape}')
    if A.ndim != 2 or A.shape[1] != 8:
        raise ValueError(f'action 必须是 (T,8)，得到 {A.shape}')
    if S.shape[0] != A.shape[0]:
        raise ValueError('state/action 长度不一致')
    g_action = gripper_from_action(A)
    starts = list(range(0, S.shape[0] - horizon + 1, stride))
    if not starts:
        raise ValueError(f'序列太短：{S.shape[0]} < horizon {horizon}')
    out = np.zeros((len(starts), horizon, 8), dtype=np.float64)
    for i, t in enumerate(starts):
        out[i, :, :7] = A[t:t + horizon, :7] - S[t, :7][None, :]
        out[i, :, 7] = g_action[t:t + horizon]
    starts_arr = np.asarray(starts, dtype=np.int64)
    return {'anchored': out, 'starts': starts_arr, 'horizon': horizon, 'stride': stride,
            'version': EXPORT_VERSION + '-anchor-horizon'}


def build_target_increment_arrays(states: np.ndarray, actions: np.ndarray,
                                  horizon: int = 16, stride: int = 4) -> dict:
    """"目标增量"表示：`k=0: A[t]-S[t]`，`k>0: A[t+k]-A[t+k-1]`；部署端用 `recursive`。

    与 anchor/fixed 表示**互逆**：两者都能精确重建专家命令序列（前者按锚点直接加，
    后者逐步累加），区别只在归一化后的数值分布与模型看到的目标。审计建议把它作为
    Gate 4 的单变量对照（`target_increment` + `recursive`）。
    """
    S = np.asarray(states, dtype=np.float64)
    A = np.asarray(actions, dtype=np.float64)
    if S.ndim != 2 or S.shape[1] != 9:
        raise ValueError(f'state 必须是 (T,9)，得到 {S.shape}')
    if A.ndim != 2 or A.shape[1] != 8:
        raise ValueError(f'action 必须是 (T,8)，得到 {A.shape}')
    if S.shape[0] != A.shape[0]:
        raise ValueError('state/action 长度不一致')
    g_action = gripper_from_action(A)
    starts = list(range(0, S.shape[0] - horizon + 1, stride))
    if not starts:
        raise ValueError(f'序列太短：{S.shape[0]} < horizon {horizon}')
    out = np.zeros((len(starts), horizon, 8), dtype=np.float64)
    for i, t in enumerate(starts):
        tgt = A[t:t + horizon, :7]
        inc = np.empty_like(tgt)
        inc[0] = tgt[0] - S[t, :7]
        if horizon > 1:
            inc[1:] = tgt[1:] - tgt[:-1]
        out[i, :, :7] = inc
        out[i, :, 7] = g_action[t:t + horizon]
    return {'target_increment': out, 'starts': np.asarray(starts, dtype=np.int64),
            'horizon': horizon, 'stride': stride,
            'version': EXPORT_VERSION + '-target-increment'}


def build_observed_residual_arrays(states: np.ndarray, actions: np.ndarray,
                                   horizon: int = 16, stride: int = 4) -> dict:
    """`observed_residual[k] = A[t+k] - S[t+k]`（**只用于诊断**：递归执行不能重建专家序列）。"""
    S = np.asarray(states, dtype=np.float64)
    A = np.asarray(actions, dtype=np.float64)
    if S.shape[0] != A.shape[0]:
        raise ValueError('state/action 长度不一致')
    g_action = gripper_from_action(A)
    starts = list(range(0, S.shape[0] - horizon + 1, stride))
    if not starts:
        raise ValueError(f'序列太短：{S.shape[0]} < horizon {horizon}')
    out = np.zeros((len(starts), horizon, 8), dtype=np.float64)
    for i, t in enumerate(starts):
        out[i, :, :7] = A[t:t + horizon, :7] - S[t:t + horizon, :7]
        out[i, :, 7] = g_action[t:t + horizon]
    return {'observed_residual': out, 'starts': np.asarray(starts, dtype=np.int64),
            'horizon': horizon, 'stride': stride,
            'version': EXPORT_VERSION + '-observed-residual'}


def build_training_arrays(states: np.ndarray, actions: np.ndarray) -> dict:
    """返回可直接喂给 π₀.₅ 训练 transform 的 8 维 state / 8 维 action（**单步**语义）。"""
    S = np.asarray(states, dtype=np.float64)
    A = np.asarray(actions, dtype=np.float64)
    if S.ndim != 2 or S.shape[1] != 9:
        raise ValueError(f'state 必须是 (T,9)，得到 {S.shape}')
    if A.ndim != 2 or A.shape[1] != 8:
        raise ValueError(f'action 必须是 (T,8)，得到 {A.shape}')
    if S.shape[0] != A.shape[0]:
        raise ValueError('state/action 长度不一致')
    dq = A[:, :7] - S[:, :7]
    g_state = gripper_from_state(S)
    g_action = gripper_from_action(A)
    state8 = np.concatenate([S[:, :7], g_state[:, None]], axis=1)
    action8 = np.concatenate([dq, g_action[:, None]], axis=1)
    action8_bin = action8.copy()
    action8_bin[:, 7] = (g_action > GRIPPER_BINARY_THRESHOLD).astype(np.float64)
    return {'state': state8, 'action': action8, 'action_binary': action8_bin,
            'gripper_state': g_state, 'gripper_action': g_action,
            'dq': dq, 'version': EXPORT_VERSION}


def audit_gripper_semantics(states: np.ndarray, actions: np.ndarray) -> dict:
    """抽查"闭合事件"：动作夹爪必须先于观测夹爪下降，且差值应显著。"""
    t = build_training_arrays(states, actions)
    gs, ga = t['gripper_state'], t['gripper_action']
    diff = ga - gs
    closing = ga < 0.6            # 专家下发闭合目标
    n_closing = int(closing.sum())
    return {
        'version': EXPORT_VERSION,
        'n_frames': int(gs.size),
        'gripper_state_mean': float(gs.mean()), 'gripper_action_mean': float(ga.mean()),
        'gripper_state_frac_open': float((gs > 0.5).mean()),
        'gripper_action_frac_open': float((ga > 0.5).mean()),
        'n_closing_frames': n_closing,
        'closing_frames_where_action_leads_state': int((diff[closing] < -0.05).sum()),
        'max_action_minus_state': float(diff.max()),
        'min_action_minus_state': float(diff.min()),
        'mean_abs_action_minus_state': float(np.abs(diff).mean()),
        'action_gripper_levels': [round(float(v), 4) for v in np.unique(np.round(ga, 3))[:8]],
        'state_gripper_levels': [round(float(v), 4) for v in np.unique(np.round(gs, 3))[:8]],
    }


def to_droid_15hz(action8: np.ndarray, *, source_hz: float = 20.0,
                   target_hz: float = 15.0) -> np.ndarray:
    """把 20 Hz 的关节增量换算到 15 Hz。

    保持**关节速度**不变：一步时间从 1/20 s 变成 1/15 s，因此
        Δq_15 = Δq_20 × (20/15) = Δq_20 × 1.3333
    （review #2：之前文档写成 ×15/20 是把比例写反了。）
    注意：这只在"原始轨迹近似匀速"时成立；更稳的做法是选定单一频率训练与推理，
    或按时间戳重采样并重新验证执行轨迹（见报告 §RQ7 修订）。
    """
    a = np.asarray(action8, dtype=np.float64).copy()
    a[:, :7] *= (source_hz / target_hz)
    return a
