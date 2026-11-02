"""实验 4.1 原子专家：参数化 A90 放置目标与 B90 墙面目标（含横向修正）。

- 只做原子技能：A 不包含任何 B 的动作，B 不包含任何 A 的历史。
- 全部动作经关节电机与真实接触完成；不重置物体、不施加约束、不瞬移。
- 继承实验 2 的 `RecordingExpert`，因此可直接录成 20 Hz 的 (state, action) 序列。
"""
from __future__ import annotations

import math

import numpy as np

from manipulation import finger_contact_forces, finger_widths, set_gripper
from vla.atomic_tasks.experts import (
    CLOSE_WIDTH, FORCE, OPEN_WIDTH, PUSH_WIDTH, PUSH_Z, SPEED, SkillExpert, axis_error_diff,
)
from vla.atomic_tasks.recording import RecordingExpert
from vla.atomic_tasks.scene2 import BLOCK_HALF, GRASP_Z, WALL_INNER_X, block_yaw, grasp_quat

FINGER_FRONT = 0.0145   # 指面前沿在 EE 前方 1.45 cm（沿用实验 2 的几何标定）


class A90GoalExpert(RecordingExpert):
    """A90：抓取 → 抬升 → 旋转 90° → 放置到**指定目标** → 释放 → 撤离。

    `run_goal` 的开关只用于评价器正负例构造（`scripts/exp41/evaluator_tests.py`），
    正式采集一律使用默认参数。
    """

    def run_goal(self, goal, *, skip_grasp: bool = False, lift_z: float = 0.15,
                 place_offset=(0.0, 0.0), force_place_yaw: float | None = None,
                 release: bool = True, retreat: bool = True,
                 drop_from: float | None = None, drop_mid: bool = False) -> dict:
        start_pos = np.asarray(self._obj()['position'], dtype=float)
        self.metrics['goal'] = {'x': goal.x, 'y': goal.y, 'goal_id': goal.goal_id}
        self.metrics['start_pos'] = start_pos.tolist()

        if not skip_grasp:
            self._grip(OPEN_WIDTH)
            self._settle(0.3)
            if not self._move((start_pos[0], start_pos[1], 0.15), math.pi / 2, 'pre_grasp').success:
                return self._result('A90', False, 'pre_grasp')
            if not self._move((start_pos[0], start_pos[1], GRASP_Z), math.pi / 2, 'grasp').success:
                return self._result('A90', False, 'grasp')
            self._grip(CLOSE_WIDTH)
            self._settle(0.8)
            if not self._move((start_pos[0], start_pos[1], lift_z), math.pi / 2, 'lift').success:
                return self._result('A90', False, 'lift')
            self.metrics['lift_m'] = float(self._obj()['position'][2] - start_pos[2])

        tx = float(goal.x) + float(place_offset[0])
        ty = float(goal.y) + float(place_offset[1])
        if force_place_yaw is not None:
            place_yaw = float(force_place_yaw)
        else:
            ok, reason = self._spin((start_pos[0], start_pos[1], 0.15), (tx, ty - 0.06, 0.15),
                                    math.pi / 2, 0.0, n=12, tag='rotate90')
            place_yaw = 0.0
            if not ok and reason == 'ik_joint_limits':
                ok, reason = self._spin((start_pos[0], start_pos[1], 0.15), (tx, ty - 0.06, 0.15),
                                        math.pi / 2, math.pi, n=12, tag='rotate90_rev')
                place_yaw = math.pi
            if not ok:
                return self._result('A90', False, reason)

        if not self._move((tx, ty, 0.15), place_yaw, 'place_above').success:
            return self._result('A90', False, 'place_above')
        if drop_mid:
            # 运输途中松开再闭合：物体在途中滑落（构造"抓取后滑落"负例）
            self._grip(OPEN_WIDTH)
            self._settle(0.6)
            self._grip(CLOSE_WIDTH)
            self._settle(0.4)
        down_z = GRASP_Z if drop_from is None else float(drop_from)
        if not self._move((tx, ty, down_z), place_yaw, 'place_down').success:
            return self._result('A90', False, 'place_down')
        if release:
            self._grip(OPEN_WIDTH)
            self._settle(0.8)
        if drop_from is not None:
            self._settle(0.8)      # 让物体自由落体并稳定
        if retreat:
            if not self._move((tx, ty - 0.16, 0.18), place_yaw, 'retreat').success:
                return self._result('A90', False, 'retreat')
            self._settle(0.6)
        self.metrics['place_target'] = [tx, ty]
        self.metrics['place_yaw_g'] = float(place_yaw)
        return self._result('A90', True, 'ok', extra=self._final())


class B90GoalExpert(RecordingExpert):
    """B90：把方块沿 +X 推到墙，并落在指定墙面目标区（可选先做横向修正）。"""

    def _push_segment(self, x_from, x_to, y, tag):
        n = 6
        for i in range(1, n + 1):
            x = x_from + (x_to - x_from) * i / n
            if not self._move((x, y, PUSH_Z), yaw=0.0, tag=f'{tag}_{i}').success:
                return False, f'{tag}_{i}'
            self._settle(0.25)
        return True, 'ok'

    def _lateral(self, pos, y_target, depth_x):
        """沿 ±Y 把方块推到 y_target（用 yaw=+90°，此时指面法向沿 Y）。"""
        dy = float(y_target - pos[1])
        if abs(dy) < 0.004:
            return True, 'ok'
        sign = 1.0 if dy > 0 else -1.0
        depth_y = BLOCK_HALF[0]          # 长轴沿 Y 时，Y 向半宽 = 2 cm
        approach = (pos[0], pos[1] - sign * 0.10, 0.07)
        if not self._move(approach, yaw=math.pi / 2, tag='b_lat_approach').success:
            return False, 'b_lat_approach'
        if not self._move((pos[0], pos[1] - sign * 0.10, PUSH_Z), yaw=math.pi / 2,
                          tag='b_lat_lower').success:
            return False, 'b_lat_lower'
        contact_y = pos[1] - sign * (depth_y + FINGER_FRONT + 0.010)
        if not self._move((pos[0], contact_y, PUSH_Z), yaw=math.pi / 2, tag='b_lat_contact').success:
            return False, 'b_lat_contact'
        self._settle(0.3)
        target_y = y_target - sign * (depth_y + FINGER_FRONT + 0.001)
        n = 5
        for i in range(1, n + 1):
            yy = contact_y + (target_y - contact_y) * i / n
            if not self._move((pos[0], yy, PUSH_Z), yaw=math.pi / 2, tag=f'b_lat_push_{i}').success:
                return False, f'b_lat_push_{i}'
            self._settle(0.25)
        self._move((pos[0], target_y - sign * 0.02, 0.08), yaw=math.pi / 2, tag='b_lat_retreat')
        self._settle(0.4)
        return True, 'ok'

    def _push_until_contact(self, pos, depth, max_extra: int = 10, step_x: float = 0.006):
        """继续以 6 mm 小步推进，直到方块与墙的间隙 ≤ 1 mm（真实接触）。

        旧版 B 专家把目标定为离墙 4 mm，因此**从来没有真正碰墙**；
        严格评价器（4.1）要求真实接触，这里用测得的物体位置闭环推进。
        物体运动仍然完全来自接触，没有 teleport/约束。
        """
        for k in range(max_extra):
            st = self._obj()
            x = float(st['position'][0])
            gap = float(WALL_INNER_X - (x + depth))
            self.metrics.setdefault('push_until_contact_gaps', []).append(round(gap, 5))
            if gap <= 0.001:
                return True, 'ok'
            ee = np.asarray(self.sim.pose()[0], dtype=float)
            target_x = float(ee[0]) + min(gap + 0.002, step_x)
            if not self._move((target_x, pos[1], PUSH_Z), yaw=0.0,
                              tag=f'b_touch_{k + 1}').success:
                return False, f'b_touch_{k + 1}'
            self._settle(0.20)
        st = self._obj()
        gap = float(WALL_INNER_X - (float(st['position'][0]) + depth))
        self.metrics['push_until_contact_final_gap_m'] = round(gap, 5)
        return (gap <= 0.002), ('ok' if gap <= 0.002 else 'wall_contact_not_reached')

    def run_goal(self, goal, *, allow_lateral: bool = True, force_wall_contact: bool = True,
                 push_z: float = PUSH_Z, direction: int = 1, drag_back: bool = False,
                 overshoot: float = 0.0) -> dict:
        st = self._obj()
        pos = np.asarray(st['position'], dtype=float)
        yaw0 = self._yaw()
        depth = BLOCK_HALF[0] * abs(math.cos(yaw0)) + BLOCK_HALF[1] * abs(math.sin(yaw0))
        gap0 = float(WALL_INNER_X - (pos[0] + depth))
        self.metrics.update({'front_gap_m': gap0, 'goal_id': goal.goal_id,
                             'start_pos': pos.tolist(), 'start_yaw': float(yaw0)})
        if gap0 <= 0.005:
            return self._result('B90', False, 'initial_too_close_to_wall')

        y_t = float(np.clip(pos[1], goal.y_lo + 0.01, goal.y_hi - 0.01))
        need_lateral = abs(y_t - pos[1]) > 0.004
        if need_lateral:
            if not allow_lateral:
                return self._result('B90', False, 'goal_requires_lateral_shift')
            ok, reason = self._lateral(pos, y_t, depth)
            if not ok:
                return self._result('B90', False, reason)
            st = self._obj()
            pos = np.asarray(st['position'], dtype=float)
            depth = BLOCK_HALF[0] * abs(math.cos(self._yaw())) + \
                BLOCK_HALF[1] * abs(math.sin(self._yaw()))
            self.metrics['after_lateral_pos'] = pos.tolist()

        self._grip(PUSH_WIDTH)
        self._settle(0.3)
        d = float(direction)
        if not self._move((pos[0] - d * 0.10, pos[1], 0.07), yaw=0.0, tag='b_approach').success:
            return self._result('B90', False, 'b_approach')
        if not self._move((pos[0] - d * 0.10, pos[1], push_z), yaw=0.0, tag='b_lower').success:
            return self._result('B90', False, 'b_lower')
        rear0 = float(pos[0] - depth)
        contact_x = rear0 - d * (FINGER_FRONT + 0.010)
        if not self._move((contact_x, pos[1], push_z), yaw=0.0, tag='b_contact').success:
            return self._result('B90', False, 'b_contact')
        self._settle(0.3)
        block_target = float(WALL_INNER_X - depth - 0.004)
        push_x = float(block_target - depth - FINGER_FRONT - 0.001)
        if d < 0:
            push_x = float(pos[0] - 0.06)      # 反方向推 6 cm（错误方向负例）
        else:
            push_x += float(overshoot)
        ok, reason = self._push_segment(contact_x, push_x, pos[1], 'b_push')
        if not ok:
            return self._result('B90', False, reason)
        if force_wall_contact and d > 0:
            ok, reason = self._push_until_contact(pos, depth)
            if not ok:
                return self._result('B90', False, reason)
        self._settle(0.5)
        if drag_back:
            # 接触后快速后退，借摩擦把方块从墙上拖开（构造"碰过又离开"的负例）
            ee = np.asarray(self.sim.pose()[0], dtype=float)
            self._move((float(ee[0]) - 0.05, pos[1], push_z), yaw=0.0, tag='b_drag_back')
            self._settle(0.6)
        if not self._move((push_x - 0.02, pos[1], 0.08), yaw=0.0, tag='b_retreat').success:
            return self._result('B90', False, 'b_retreat')
        self._settle(0.6)
        final_state = self._obj()
        final = self._final()
        self.metrics.update({
            'front_gap_end_m': float(WALL_INNER_X - (final_state['position'][0] + depth)),
            'pushed_x_m': float(final_state['position'][0] - pos[0]),
            'lateral_dy_m': float(final_state['position'][1] - pos[1]),
        })
        return self._result('B90', True, 'ok', extra=final)
