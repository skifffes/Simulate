"""安全约束：任何一条不满足就停止，并把原因写进日志。

prompt Step 7.4 要求最小测试必须具备：
    关节位置限制 / EE workspace 限制 / 单步动作幅度限制 / 不合法动作拒绝 /
    NaN·Inf 检测 / 非预期碰撞检测 / 物体异常运动停止 / 控制器异常停止 / 超时退出

本模块只做“判定 + 记录”，不修改动作本身（动作的裁剪发生在 action_adapter，且带理由）。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np

STOP_REASONS = {
    'nan_or_inf_action': '策略动作含 NaN/Inf',
    'rejected_action': '动作被适配器拒绝（超官方 clip / 关节限位）',
    'joint_limit': '关节超出 URDF 限位',
    'joint_step_too_large': '单个控制步的关节变化超过上限',
    'ee_workspace': '末端位置超出安全工作空间',
    'unexpected_collision': '检测到非预期碰撞',
    'object_abnormal': '物体出现异常运动（速度/位置越界）',
    'controller_tracking': '控制器跟踪误差过大',
    'timeout': '超时',
    'max_ticks': '达到设定的最大控制步数',
    'server_error': '策略服务器错误',
}


@dataclass
class SafetyLimits:
    """全部显式写进工件，便于复现与跨实验比较。"""

    joint_margin_rad: float = 0.02
    max_joint_step_rad: float = 0.30      # 一个控制步内任一关节的最大变化
    ee_x: tuple = (0.15, 0.95)
    ee_y: tuple = (-0.45, 0.60)
    ee_z: tuple = (0.005, 0.60)
    object_speed_max: float = 3.0         # m/s，超过即认为异常
    object_z_min: float = -0.05
    tracking_error_max: float = 0.25      # rad，目标与实际关节的 L2
    wall_time_max_s: float = 120.0
    max_ticks: int = 250
    # 预期接触（link 对，顺序无关）：两个指尖互碰。夹爪闭合到 0 时必然发生，
    # 这是实验 2/3 里也没有被当成失败的接触（simulation.collisions() 只排除了父子对）。
    expected_contact_pairs: tuple = ((9, 10),)


@dataclass
class SafetyEvent:
    tick: int
    code: str
    detail: dict = field(default_factory=dict)
    wall_time: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        return {'tick': self.tick, 'code': self.code,
                'message': STOP_REASONS.get(self.code, self.code),
                'detail': self.detail, 'wall_time': self.wall_time}


class SafetyMonitor:
    def __init__(self, sim, limits: SafetyLimits | None = None, object_ids=()):
        self.sim = sim
        self.limits = limits or SafetyLimits()
        self.object_ids = tuple(object_ids)
        self.events: list[SafetyEvent] = []
        self.tick = 0
        self.t0 = time.time()
        self.stopped = False
        self.expected_contacts_seen = 0
        self._prev_q = np.asarray(sim.q(), dtype=float)

    # ---- 记录 -------------------------------------------------------------
    def record(self, code: str, **detail) -> SafetyEvent:
        ev = SafetyEvent(self.tick, code, detail)
        self.events.append(ev)
        self.stopped = True
        return ev

    def why_stopped(self) -> str | None:
        return self.events[-1].code if self.events else None

    # ---- 检查 -------------------------------------------------------------
    def check_action(self, action: np.ndarray, adapted_reasons=()) -> bool:
        a = np.asarray(action, dtype=float)
        if not np.all(np.isfinite(a)):
            self.record('nan_or_inf_action')
            return False
        if adapted_reasons:
            self.record('rejected_action', reasons=list(adapted_reasons))
            return False
        return True

    def check_state(self, q_prev: np.ndarray, q_now: np.ndarray, lower, upper) -> bool:
        q_prev = np.asarray(q_prev, dtype=float)
        q_now = np.asarray(q_now, dtype=float)
        m = self.limits.joint_margin_rad
        if np.any(q_now < lower - m) or np.any(q_now > upper + m):
            self.record('joint_limit', q=q_now.tolist(),
                        lower=lower.tolist(), upper=upper.tolist())
            return False
        step = float(np.max(np.abs(q_now - q_prev)))
        if step > self.limits.max_joint_step_rad:
            self.record('joint_step_too_large', step_rad=step,
                        limit=self.limits.max_joint_step_rad)
            return False
        return True

    def check_ee(self, position) -> bool:
        x, y, z = (float(v) for v in position)
        lim = self.limits
        ok = (lim.ee_x[0] <= x <= lim.ee_x[1] and lim.ee_y[0] <= y <= lim.ee_y[1]
              and lim.ee_z[0] <= z <= lim.ee_z[1])
        if not ok:
            self.record('ee_workspace', position=[x, y, z],
                        box={'x': lim.ee_x, 'y': lim.ee_y, 'z': lim.ee_z})
        return ok

    def check_collisions(self, allowed=None) -> bool:
        bad = self.sim.collisions(allowed=allowed) if allowed is not None else self.sim.collisions()
        if bad:
            # 过滤掉预期接触（指尖互碰），并把被过滤的条数记进事件里，避免“静默通过”
            expected = {frozenset(pair) for pair in self.limits.expected_contact_pairs}
            kept, filtered = [], 0
            for c in bad:
                pair = frozenset((c.get('link_a'), c.get('link_b')))
                if c.get('body') == self.sim.robot and pair in expected:
                    filtered += 1
                    continue
                kept.append(c)
            if kept:
                self.record('unexpected_collision', contacts=kept[:3], filtered_expected=filtered)
                return False
            self.expected_contacts_seen += filtered
        return True

    def check_objects(self, object_ids=None) -> bool:
        import pybullet as p

        ids = self.object_ids if object_ids is None else tuple(object_ids)
        if not ids:
            return True
        for oid in ids:
            lin, _ = p.getBaseVelocity(oid, physicsClientId=self.sim.client)
            pos, _ = p.getBasePositionAndOrientation(oid, physicsClientId=self.sim.client)
            speed = float(np.linalg.norm(lin))
            if speed > self.limits.object_speed_max:
                self.record('object_abnormal', body=oid, speed=speed, position=list(pos))
                return False
            if float(pos[2]) < self.limits.object_z_min:
                self.record('object_abnormal', body=oid, reason='below_table', position=list(pos))
                return False
        return True

    def check_tracking(self, q_target: np.ndarray, q_now: np.ndarray) -> bool:
        err = float(np.linalg.norm(np.asarray(q_now) - np.asarray(q_target)))
        if err > self.limits.tracking_error_max:
            self.record('controller_tracking', error_rad=err, limit=self.limits.tracking_error_max)
            return False
        return True

    def check_time(self) -> bool:
        elapsed = time.time() - self.t0
        if elapsed > self.limits.wall_time_max_s:
            self.record('timeout', elapsed_s=elapsed, limit=self.limits.wall_time_max_s)
            return False
        return True

    def tick_once(self) -> bool:
        self.tick += 1
        if self.tick > self.limits.max_ticks:
            self.record('max_ticks', ticks=self.tick)
            return False
        return True

    # ---- 导出 -------------------------------------------------------------
    def summary(self) -> dict:
        return {
            'stopped': self.stopped,
            'stop_reason': self.why_stopped(),
            'ticks': self.tick,
            'wall_time_s': round(time.time() - self.t0, 3),
            'expected_contacts_seen': self.expected_contacts_seen,
            'limits': {
                'joint_margin_rad': self.limits.joint_margin_rad,
                'max_joint_step_rad': self.limits.max_joint_step_rad,
                'ee_box': {'x': list(self.limits.ee_x), 'y': list(self.limits.ee_y),
                           'z': list(self.limits.ee_z)},
                'object_speed_max': self.limits.object_speed_max,
                'tracking_error_max': self.limits.tracking_error_max,
                'wall_time_max_s': self.limits.wall_time_max_s,
                'max_ticks': self.limits.max_ticks,
            },
            'events': [e.as_dict() for e in self.events],
        }

    def dump(self, path) -> None:
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(self.summary(), fh, ensure_ascii=False, indent=2, default=float)
