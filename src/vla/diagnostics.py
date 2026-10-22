"""闭环评测的逐 tick 观测与失败诊断。

这里的所有量都来自 PyBullet 真值与接触数据，只用于离线分析，
绝不会进入策略输入（策略只看两路 RGB + 9 维本体状态 + 固定指令）。
"""
from __future__ import annotations

import numpy as np
import pybullet as p

from manipulation import FINGER_LINKS, contact_points, finger_tips, relative_pose

# 阶段事件阈值（米 / 秒 / 弧度）
LIFTOFF_DZ = 0.005
LIFT_TARGET_DZ = 0.08
TARGET_ENTRY_XY = 0.03
RELEASE_WIDTH = 0.030
CLOSE_WIDTH = 0.025  # 工件半宽 0.02 m，闭合后单指指令约 0.02
STABLE_SECONDS = 0.30
CLOSE_GRACE_TICKS = 20


def _xy(a, b):
    return float(np.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


class EpisodeMonitor:
    """记录 rollout 的物理真值，抽取阶段事件并给出主要失败原因。"""

    def __init__(self, sim, block, task, dt: float):
        self.sim, self.block, self.task = sim, block, task
        self.dt = float(dt)
        self.init = np.asarray(task.initial['position'], dtype=float)
        self.rows = []
        self.events = {}
        self.max_z = float(self.init[2])
        self.slip = 0.0
        self._grasp_local = None
        self._last_action = None
        self._prev_sign = None
        self._action_flips = 0
        self._action_pairs = 0
        self._tracking_err = []
        self._saturated = 0
        self._clipped = 0
        self._contact_ticks = 0
        self._release_free_tick = None
        self._stable_since = None
        self._stable_tick = None

    # ------------------------------------------------------------------
    def tick(self, tick: int, action: np.ndarray, target: np.ndarray, width_cmd: float,
             q: np.ndarray, obj: dict, phase: str = 'policy'):
        sim = self.sim
        state_ee = p.getLinkState(sim.robot, 8, computeForwardKinematics=True,
                                  physicsClientId=sim.client)
        ee = np.asarray(state_ee[4], dtype=float)
        tips = np.asarray(finger_tips(sim), dtype=float)
        center = tips.mean(axis=0)
        obj_pos = np.asarray(obj['position'], dtype=float)

        contacts = contact_points(sim, self.block)
        f_finger = [sum(c['normal_force'] for c in contacts if c['link'] == link)
                    for link in FINGER_LINKS]
        f_hand = sum(c['normal_force'] for c in contacts
                     if c['link'] not in FINGER_LINKS)
        dual = bool(f_finger[0] > 0.0 and f_finger[1] > 0.0)
        if dual:
            self._contact_ticks += 1

        q = np.asarray(q, dtype=float)
        target = np.asarray(target, dtype=float)
        track = float(np.max(np.abs(q[:7] - target[:7])))
        self._tracking_err.append(track)
        if track > 0.05:
            self._saturated += 1
        if np.any(np.asarray(action[:7]) < np.asarray(sim.lower) + 1e-4) or \
                np.any(np.asarray(action[:7]) > np.asarray(sim.upper) - 1e-4):
            self._clipped += 1
        if self._last_action is not None:
            d = np.asarray(action[:7], dtype=float) - self._last_action
            s = np.where(d > 1e-6, 1, np.where(d < -1e-6, -1, 0))
            if self._prev_sign is not None:
                both = (s != 0) & (self._prev_sign != 0)
                self._action_flips += int(np.sum((s != self._prev_sign) & both))
                self._action_pairs += 1
            self._prev_sign = s
        self._last_action = np.asarray(action[:7], dtype=float).copy()

        if self._grasp_local is None and (obj_pos[2] - self.init[2]) > 0.02:
            self._grasp_local = relative_pose(sim, self.block)
        if self._grasp_local is not None:
            local = relative_pose(sim, self.block)
            self.slip = max(self.slip, float(np.linalg.norm(
                np.asarray(local[0]) - np.asarray(self._grasp_local[0]))))
        self.max_z = max(self.max_z, float(obj_pos[2]))

        width_actual = float(np.mean([
            s[0] for s in p.getJointStates(sim.robot, list(FINGER_LINKS),
                                          physicsClientId=sim.client)]))
        self.rows.append({
            'tick': int(tick), 'phase': phase, 't': float(tick) * self.dt,
            'obj_x': float(obj_pos[0]), 'obj_y': float(obj_pos[1]), 'obj_z': float(obj_pos[2]),
            'ee_x': float(ee[0]), 'ee_y': float(ee[1]), 'ee_z': float(ee[2]),
            'grip_x': float(center[0]), 'grip_y': float(center[1]), 'grip_z': float(center[2]),
            'width_cmd': float(width_cmd), 'width_actual': width_actual,
            'f_finger_l': float(f_finger[0]), 'f_finger_r': float(f_finger[1]),
            'f_hand': float(f_hand), 'dual_contact': int(dual),
            'obj_speed': float(obj['linear_speed']),
            'tracking_err': track,
        })

        ev = self.events
        if 'approach' not in ev and _xy(center, obj_pos) < 0.05 and center[2] > obj_pos[2] + 0.03:
            ev['approach'] = tick
        if 'first_contact' not in ev and (f_finger[0] > 0 or f_finger[1] > 0):
            ev['first_contact'] = tick
        if 'close_cmd' not in ev and width_cmd <= CLOSE_WIDTH and tick > 2:
            ev['close_cmd'] = tick
            ev['close_offset_xy'] = _xy(center, obj_pos)
            ev['close_offset_z'] = float(center[2] - obj_pos[2])
            ev['close_width_cmd'] = float(width_cmd)
            ev['close_width_actual'] = width_actual
            ev['close_dual_contact'] = int(dual)
        if 'close_cmd' in ev and 'dual_contact' not in ev and tick - ev['close_cmd'] <= CLOSE_GRACE_TICKS:
            if dual:
                ev['dual_contact'] = tick
        if 'liftoff' not in ev and obj_pos[2] - self.init[2] > LIFTOFF_DZ:
            ev['liftoff'] = tick
            ev['liftoff_offset'] = float(np.linalg.norm(obj_pos - center))
        if 'lift_target' not in ev and obj_pos[2] - self.init[2] > LIFT_TARGET_DZ:
            ev['lift_target'] = tick
        if 'target_entry' not in ev and _xy(obj_pos, self.task.params['place']) <= TARGET_ENTRY_XY \
                and obj_pos[2] > self.task.params['block_half'][2] + 0.02:
            ev['target_entry'] = tick
            ev['target_entry_xy'] = _xy(obj_pos, self.task.params['place'])
        if 'release_cmd' not in ev and 'close_cmd' in ev and width_cmd >= RELEASE_WIDTH:
            ev['release_cmd'] = tick
        if 'release_cmd' in ev and self._release_free_tick is None and dual is False \
                and f_finger[0] <= 0.0 and f_finger[1] <= 0.0:
            self._release_free_tick = tick
            ev['release_free'] = tick
        if 'release_cmd' in ev and phase == 'settle':
            if float(obj['linear_speed']) < 0.01:
                if self._stable_since is None:
                    self._stable_since = (tick, float(tick) * self.dt)
                elif float(tick) * self.dt - self._stable_since[1] >= STABLE_SECONDS:
                    self._stable_tick = tick
                    ev['stable'] = tick
            else:
                self._stable_since = None

    # ------------------------------------------------------------------
    def summary(self, result: dict, timeout: bool, mode: str) -> dict:
        ev = self.events
        init_z = float(self.init[2])
        lift = float(result.get('lift_height_m', self.max_z - init_z))
        xy_err = float(result.get('place_xy_error_m', float('nan')))
        released = 'release_cmd' in ev
        final_width = self.rows[-1]['width_actual'] if self.rows else 0.0

        reasons = []
        if not result.get('success'):
            if 'close_cmd' not in ev:
                primary, stage = 'no_gripper_close', 'approach'
                reasons.append('全程未下达闭合指令')
            elif not ev.get('close_dual_contact', 0):
                primary, stage = 'no_grasp_contact', 'grasp'
                reasons.append('闭爪后双指未同时接触工件')
            elif lift < 0.03:
                primary, stage = 'lift_failed', 'lift'
                reasons.append(f'抬升仅 {lift * 1000:.0f} mm')
            elif self.slip > 0.02:
                primary, stage = 'dropped_in_transport', 'transport'
                reasons.append(f'相对滑移 {self.slip * 1000:.0f} mm')
            elif 'target_entry' not in ev and not released:
                primary, stage = 'no_autonomous_release', 'release'
                reasons.append('未自主张开夹爪')
            elif 'target_entry' not in ev:
                primary, stage = 'place_position_error', 'place'
                reasons.append(f'放置点偏差 {xy_err * 1000:.0f} mm')
            elif not released:
                primary, stage = 'no_autonomous_release', 'release'
                reasons.append('到达目标区但未自主张开夹爪')
            elif result.get('release_contacts', 0) > 0:
                primary, stage = 'no_autonomous_release', 'release'
                reasons.append('释放后仍与工件接触')
            elif timeout:
                primary, stage = 'timeout', 'timeout'
                reasons.append('超时')
            else:
                primary, stage = (result.get('failure') or 'other'), 'settle'
                reasons.append(str(result.get('failure_text') or result.get('failure')))
        else:
            primary, stage = 'completed', 'completed'

        if ev.get('close_cmd') is not None and ev.get('first_contact') is None:
            reasons.append('闭爪时尚未接触工件')
        if ev.get('close_offset_xy', 0) > 0.02:
            reasons.append(f"闭爪时水平偏差 {ev['close_offset_xy'] * 1000:.0f} mm")
        if self._saturated > 15:
            reasons.append(f'{self._saturated} 个 tick 跟踪误差 > 0.05 rad')

        return {
            'mode': mode,
            'failure_primary': primary,
            'failure_stage_event': stage,
            'failure_reasons': '; '.join(dict.fromkeys(reasons)),
            'autonomous_release': bool(released),
            'release_tick': ev.get('release_cmd'),
            'first_contact_tick': ev.get('first_contact'),
            'close_tick': ev.get('close_cmd'),
            'liftoff_tick': ev.get('liftoff'),
            'lift_target_tick': ev.get('lift_target'),
            'target_entry_tick': ev.get('target_entry'),
            'release_free_tick': ev.get('release_free'),
            'stable_tick': ev.get('stable'),
            'close_offset_xy_m': float(ev.get('close_offset_xy', float('nan'))),
            'close_offset_z_m': float(ev.get('close_offset_z', float('nan'))),
            'close_width_cmd': float(ev.get('close_width_cmd', float('nan'))),
            'close_width_actual': float(ev.get('close_width_actual', float('nan'))),
            'close_dual_contact': int(ev.get('close_dual_contact', 0)),
            'dual_contact_ticks': int(self._contact_ticks),
            'slip_m': float(self.slip),
            'max_obj_z': float(self.max_z),
            'final_width_actual': float(final_width),
            'saturated_ticks': int(self._saturated),
            'clipped_action_ticks': int(self._clipped),
            'max_tracking_err_rad': float(np.max(self._tracking_err) if self._tracking_err else 0.0),
            'mean_tracking_err_rad': float(np.mean(self._tracking_err) if self._tracking_err else 0.0),
            'action_sign_flips_per_tick': float(self._action_flips / max(self._action_pairs, 1)),
            'timeout': bool(timeout),
        }


__all__ = ['EpisodeMonitor']
