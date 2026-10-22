"""实验 2 的原子技能专家：A0 / A90 / B / C。

专家只用于物理可行性检查、示范采集和评测基线；VLA 评测时绝不参与控制。
所有动作都通过关节电机与真实接触完成：不重置物体、不施加固定约束、不瞬移。
"""
from __future__ import annotations

import math
import numpy as np
import pybullet as p

from manipulation import (
    finger_contact_forces, object_state, relative_pose, set_gripper,
)
from robot_controller import RobotController

from .scene2 import (
    BLOCK_HALF, GRASP_Z, PLACE, START, TRAY, TRAY_INNER_GAP, WALL_INNER_X,
    block_yaw, grasp_quat,
)

OPEN_WIDTH = 0.04
CLOSE_WIDTH = 0.017
PUSH_WIDTH = 0.014   # 推时不完全闭合，避免两指互碰（自碰撞会被判成失败）
PUSH_Z = 0.018       # 推到物体半高以下，避免把物体顶翻/爬上指面
FORCE = 20.0
SPEED = 0.06


def axis_error_y(yaw: float) -> float:
    """长轴沿 Y 的误差（rad，0 最好）。"""
    a = abs(math.atan2(math.sin(yaw), math.cos(yaw))) % math.pi
    return float(min(a, math.pi - a))


def axis_error_x(yaw: float) -> float:
    """长轴沿 X 的误差（rad，0 最好）。"""
    a = abs(math.atan2(math.sin(yaw), math.cos(yaw))) % math.pi
    return float(abs(a - math.pi / 2))


class SkillExpert:
    def __init__(self, sim, block):
        self.sim, self.block = sim, block
        self.ctl = RobotController(sim)
        self.trace = []
        self.metrics = {}

    # ---------------- 基本原语 ----------------
    def _settle(self, seconds: float):
        for _ in range(max(1, int(round(seconds / self.sim.dt)))):
            self.sim.step()

    def _grip(self, width: float):
        set_gripper(self.sim, width, FORCE, SPEED)

    def _obj(self):
        return object_state(self.sim, self.block)

    def _yaw(self):
        return block_yaw(self.block, self.sim.client)

    def _move(self, pos, yaw, tag):
        target = np.asarray(pos, dtype=float)
        result = self.ctl.move_pose(target, grasp_quat(yaw))
        entry = {'tag': tag, 'target': target.tolist(), 'yaw': float(yaw),
                 'ok': bool(result.success), 'reason': result.reason}
        if not result.success and result.reason == 'collision_during_motion':
            entry['collisions'] = self.sim.collisions()
            st = self._obj()
            entry['obj'] = {'pos': st['position'].tolist(), 'tilt': st['tilt_deg']}
            entry['link6'] = [float(v) for v in
                              p.getLinkState(self.sim.robot, 6,
                                             computeForwardKinematics=True,
                                             physicsClientId=self.sim.client)[4]]
        self.trace.append(entry)
        return result

    def _move_joints(self, q, tag):
        result = self.ctl.move(np.asarray(q, dtype=float))
        self.trace.append({'tag': tag, 'ok': bool(result.success), 'reason': result.reason})
        return result

    def _spin(self, p0, p1, yaw0, yaw1, n=10, tag='spin'):
        """在保持夹持的同时，把末端位置与 yaw 一起插值到目标。"""
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        waypoints, prev = [], self.sim.q()
        for i in range(1, n + 1):
            a = i / n
            pos = p0 + (p1 - p0) * a
            yaw = yaw0 + (yaw1 - yaw0) * a
            q, ok, _ = self.ctl.solve(pos, grasp_quat(yaw), rest=prev)
            if not ok:
                return False, 'ik_joint_limits'
            waypoints.append(q)
            prev = q
        for q in waypoints:
            result = self.ctl.move(q, sim_steps=600)
            if not result.success:
                return False, result.reason
        self.trace.append({'tag': tag, 'yaw0': float(yaw0), 'yaw1': float(yaw1), 'ok': True})
        return True, 'ok'

    # ---------------- 抓取与放置 ----------------
    def pick(self, yaw_g: float, lift_z: float = 0.15):
        pos = self._obj()['position']
        self._grip(OPEN_WIDTH)
        self._settle(0.3)
        if not self._move((pos[0], pos[1], 0.15), yaw_g, 'pre_grasp').success:
            return False, 'pre_grasp'
        if not self._move((pos[0], pos[1], GRASP_Z), yaw_g, 'grasp').success:
            return False, 'grasp'
        self._grip(CLOSE_WIDTH)
        self._settle(0.8)
        f1, f2, active = finger_contact_forces(self.sim, self.block)
        if len(active) < 2:
            return False, 'grasp_no_contact'
        if not self._move((pos[0], pos[1], lift_z), yaw_g, 'lift').success:
            return False, 'lift'
        lifted = self._obj()['position'][2] - pos[2]
        self.metrics['lift_m'] = float(lifted)
        if lifted < 0.08:
            return False, 'object_not_lifted'
        return True, 'ok'

    def place(self, target, yaw_g: float, retreat_z: float = 0.18,
              open_width: float = OPEN_WIDTH):
        if not self._move((target[0], target[1], 0.15), yaw_g, 'place_above').success:
            return False, 'place_above'
        if not self._move((target[0], target[1], GRASP_Z), yaw_g, 'place_down').success:
            return False, 'place_down'
        self._grip(open_width)
        self._settle(0.8)
        if not self._move((target[0], target[1], retreat_z), yaw_g, 'retreat').success:
            return False, 'retreat'
        self._settle(0.6)
        return True, 'ok'

    def _final(self, max_speed: float = 0.01):
        state = self._obj()
        return {
            'position': state['position'].tolist(),
            'yaw_rad': self._yaw(),
            'tilt_deg': state['tilt_deg'],
            'linear_speed': state['linear_speed'],
            'rest': bool(state['linear_speed'] < max_speed),
        }

    # ---------------- 原子技能 ----------------
    def skill_a0(self):
        ok, reason = self.pick(yaw_g=math.pi / 2)
        if not ok:
            return self._result('A0', False, reason)
        ok, reason = self.place(PLACE, yaw_g=math.pi / 2)
        if not ok:
            return self._result('A0', False, reason)
        final = self._final()
        err = float(np.linalg.norm(np.asarray(final['position'][:2]) - np.asarray(PLACE[:2])))
        good = (err <= 0.02 and axis_error_y(final['yaw_rad']) <= math.radians(12)
                and final['tilt_deg'] <= 12 and final['rest'])
        return self._result('A0', good, 'ok' if good else 'place_check',
                            extra={'xy_error_m': err, **final})

    def skill_a90(self):
        start = self._obj()['position'].copy()
        ok, reason = self.pick(yaw_g=math.pi / 2)
        if not ok:
            return self._result('A90', False, reason)
        # 夹爪 yaw 从 +90° 转到 0°：物体随之从长轴 Y 转到长轴 X。
        ok, reason = self._spin((start[0], start[1], 0.15), (start[0], PLACE[1] - 0.06, 0.15),
                                math.pi / 2, 0.0, n=12, tag='rotate90')
        if not ok:
            return self._result('A90', False, reason)
        ok, reason = self.place(PLACE, yaw_g=0.0)
        if not ok:
            return self._result('A90', False, reason)
        final = self._final()
        err = float(np.linalg.norm(np.asarray(final['position'][:2]) - np.asarray(PLACE[:2])))
        good = (err <= 0.02 and axis_error_x(final['yaw_rad']) <= math.radians(12)
                and final['tilt_deg'] <= 12 and final['rest'])
        return self._result('A90', good, 'ok' if good else 'place_check',
                            extra={'xy_error_m': err, **final})

    def skill_b(self):
        state = self._obj()
        pos = np.asarray(state['position'], dtype=float)
        yaw0 = self._yaw()
        depth = BLOCK_HALF[0] * abs(math.cos(yaw0)) + BLOCK_HALF[1] * abs(math.sin(yaw0))
        gap0 = float(WALL_INNER_X - (pos[0] + depth))
        self.metrics['front_gap_m'] = gap0
        self._grip(PUSH_WIDTH)
        self._settle(0.3)
        if not self._move((pos[0] - 0.10, pos[1], 0.07), yaw=0.0, tag='b_approach').success:
            return self._result('B', False, 'b_approach')
        if not self._move((pos[0] - 0.10, pos[1], PUSH_Z), yaw=0.0, tag='b_lower').success:
            return self._result('B', False, 'b_lower')
        # 指面前沿在 EE 前方 1.45 cm、手掌前沿在 3.56 cm；推的目标要给指面留压缩量，
        # 否则会把物体顶到墙上、再被挤得翻起来。
        finger_front = 0.0145
        rear0 = float(pos[0] - depth)
        contact_x = rear0 - finger_front - 0.010
        if not self._move((contact_x, pos[1], PUSH_Z), yaw=0.0, tag='b_contact').success:
            return self._result('B', False, 'b_contact')
        self._settle(0.3)
        block_target = float(WALL_INNER_X - depth - 0.004)
        push_x = float(block_target - depth - finger_front - 0.001)
        n_seg = 6
        for i in range(1, n_seg + 1):
            x = contact_x + (push_x - contact_x) * i / n_seg
            if not self._move((x, pos[1], PUSH_Z), yaw=0.0, tag=f'b_push_{i}').success:
                return self._result('B', False, f'b_push_{i}')
            self._settle(0.25)
        self._settle(0.5)
        if not self._move((push_x - 0.02, pos[1], 0.08), yaw=0.0, tag='b_retreat').success:
            return self._result('B', False, 'b_retreat')
        final_state = self._obj()
        final = self._final()
        dy = float(final_state['position'][1] - pos[1])
        depth1 = BLOCK_HALF[0] * abs(math.cos(final['yaw_rad'])) + \
            BLOCK_HALF[1] * abs(math.sin(final['yaw_rad']))
        gap1 = float(WALL_INNER_X - (final_state['position'][0] + depth1))
        pushed = float(final_state['position'][0] - pos[0])
        self.metrics.update({'front_gap_end_m': gap1, 'pushed_x_m': pushed,
                             'lateral_dy_m': dy,
                             'yaw_change_rad': final['yaw_rad'] - yaw0})
        good = (abs(gap1) <= 0.02 and abs(dy) <= 0.02 and
                axis_error_diff(final['yaw_rad'], yaw0) <= math.radians(15) and
                final['tilt_deg'] <= 12 and final['rest'])
        return self._result('B', good, 'ok' if good else 'push_check',
                            extra={'gap_m': gap1, **final})

    def skill_c(self):
        state = self._obj()
        pos = np.asarray(state['position'], dtype=float)
        yaw0 = self._yaw()
        long_y = axis_error_y(yaw0) < math.radians(45)
        yaw_g = math.pi / 2 if long_y else 0.0
        ok, reason = self.pick(yaw_g=yaw_g)
        if not ok:
            return self._result('C', False, reason)
        final_yaw_g = yaw_g
        if not long_y:
            # 长轴 X -> 需要旋回 Y：夹爪 yaw 从 0 转到 -90°。
            ok, reason = self._spin((pos[0], pos[1], 0.15), (TRAY[0], TRAY[1], 0.15),
                                    0.0, -math.pi / 2, n=12, tag='rotate_to_tray')
            if not ok:
                return self._result('C', False, reason)
            final_yaw_g = -math.pi / 2
        ok, reason = self.place(TRAY, yaw_g=final_yaw_g, open_width=0.023)
        if not ok:
            return self._result('C', False, reason)
        final_state = self._obj()
        final = self._final()
        dx = float(final_state['position'][0] - TRAY[0])
        dy = float(final_state['position'][1] - TRAY[1])
        good = (abs(dx) <= 0.02 and abs(dy) <= 0.05 and
                axis_error_y(final['yaw_rad']) <= math.radians(12) and
                final['tilt_deg'] <= 12 and final['rest'])
        self.metrics.update({'tray_dx_m': dx, 'tray_dy_m': dy,
                             'tray_clearance_m': TRAY_INNER_GAP - 2 * BLOCK_HALF[0]})
        return self._result('C', good, 'ok' if good else 'tray_check',
                            extra={'tray_dx_m': dx, 'tray_dy_m': dy, **final})

    # ---------------- 汇总 ----------------
    def _result(self, skill, success, reason, extra=None):
        out = {'skill': skill, 'success': bool(success), 'reason': reason,
               'metrics': dict(self.metrics), 'trace': self.trace}
        if extra:
            out['final'] = extra
        return out


def axis_error_diff(yaw1: float, yaw0: float) -> float:
    d = math.atan2(math.sin(yaw1 - yaw0), math.cos(yaw1 - yaw0))
    return float(abs(d))
