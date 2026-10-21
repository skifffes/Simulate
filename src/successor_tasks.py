"""实验 0D 阶段二后继任务：B 侧面推动、C 顶抓提升。

B 用夹爪闭合后的指外侧推工件的指定局部侧面（局部 -y 面），推到挡板；
C 以固定的顶抓姿态（指缝沿世界 y）从上方闭合双指、提升并保持。
两者都只在真实接触上工作：不做物体 teleport，不切换 kinematic，不加固定约束。
"""
import numpy as np
import pybullet as p

from manipulation import (
    FINGER_MAX_FORCE, contact_points, finger_contact_forces, finger_widths,
    non_finger_contacts, object_state, relative_pose, set_gripper,
)

WALL_Y = 0.36           # 挡板内表面
WALL_HALF = (0.10, 0.01, 0.05)
PUSH_HEIGHT = 0.022
HALF = (0.05, 0.02, 0.02)


def create_wall(sim):
    shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=list(WALL_HALF),
                                   physicsClientId=sim.client)
    visual = p.createVisualShape(p.GEOM_BOX, halfExtents=list(WALL_HALF),
                                 rgbaColor=[0.35, 0.38, 0.42, 1.0],
                                 physicsClientId=sim.client)
    return p.createMultiBody(0, shape, visual,
                             basePosition=[0.45, WALL_Y + WALL_HALF[1], WALL_HALF[2]],
                             physicsClientId=sim.client)


def object_local_point(sim, body, point):
    pos, quat = p.getBasePositionAndOrientation(body, physicsClientId=sim.client)
    m = np.asarray(p.getMatrixFromQuaternion(quat), dtype=float).reshape(3, 3)
    return m.T @ (np.asarray(point, dtype=float) - np.asarray(pos, dtype=float))


def orientation_for(yaw_rad):
    return tuple(p.getQuaternionFromEuler([np.pi, 0.0, float(yaw_rad)]))


class SuccessorTask:
    """两种后继任务共用的记录、运动重试与判定基础设施。"""

    failure_text = {}

    def __init__(self, sim, controller, block, params=None, on_step=None,
                 record_interval=4, contact_interval=8):
        self.sim, self.ctl, self.block = sim, controller, block
        self.params = dict(self.defaults)
        if params:
            self.params.update(params)
        self.on_step = on_step
        self.record_interval = max(1, int(record_interval))
        self.contact_interval = max(1, int(contact_interval))
        self.samples, self.contacts, self.stages = [], [], []
        self.step0 = self.sim.steps
        # B 需要挡板；由调用方注入，缺省时在 run() 内创建。
        self.wall = None
        self.failure = self.failure_stage = None
        self.success = False
        self.hand_contact_samples = 0
        self._last_contact_sample = -10 ** 9

    defaults = {}
    params = {}

    def now(self):
        return (self.sim.steps - self.step0) * self.sim.dt

    def _sample(self, stage):
        state = object_state(self.sim, self.block)
        f1, f2, active = finger_contact_forces(self.sim, self.block)
        widths = finger_widths(self.sim)
        ee = self.sim.pose()[0]
        self.samples.append({
            't': self.now(), 'stage': stage,
            'ee_x': float(ee[0]), 'ee_y': float(ee[1]), 'ee_z': float(ee[2]),
            'obj_x': float(state['position'][0]), 'obj_y': float(state['position'][1]),
            'obj_z': float(state['position'][2]),
            'obj_lin_speed': state['linear_speed'],
            'obj_ang_speed': state['angular_speed'],
            'tilt_deg': state['tilt_deg'],
            'finger1': widths[0], 'finger2': widths[1],
            'contact_force_1': f1, 'contact_force_2': f2,
            'contact_links': len(active),
        })
        if non_finger_contacts(self.sim, self.block):
            self.hand_contact_samples += 1
        if self.sim.steps - self._last_contact_sample >= self.contact_interval:
            self._last_contact_sample = self.sim.steps
            for c in contact_points(self.sim, self.block):
                if c['normal_force'] <= 0.0:
                    continue
                local = object_local_point(self.sim, self.block, c['position'])
                self.contacts.append({
                    't': self.now(), 'stage': stage, 'link': c['link'],
                    'normal_force': c['normal_force'],
                    'px': float(c['position'][0]), 'py': float(c['position'][1]),
                    'pz': float(c['position'][2]),
                    'lx': float(local[0]), 'ly': float(local[1]), 'lz': float(local[2]),
                })

    def _sample_due(self, stage):
        if self.sim.steps % self.record_interval == 0:
            self._sample(stage)

    def _step_stage(self, stage, seconds, stop=None):
        steps = max(1, int(round(seconds / self.sim.dt)))
        start = self.sim.steps
        for _ in range(steps):
            self.sim.step()
            self._sample_due(stage)
            if self.on_step is not None:
                self.on_step(stage, self.sim.steps - start, steps)
            if stop is not None and stop():
                break
        self._sample(stage)
        self.stages.append({'stage': stage, 'success': True, 'reason': 'ok',
                            'seconds': (self.sim.steps - start) * self.sim.dt})

    def _move(self, stage, target, yaw_deg=0.0, attempts=3):
        start = self.sim.steps
        goal = np.asarray(target, dtype=float)
        result = None
        for attempt in range(attempts):
            def observer(k, n, q, seconds):
                self._sample_due(stage)
                if self.on_step is not None and k % 4 == 0:
                    self.on_step(stage, k, n + 60)

            result = self.ctl.move_pose(goal, orientation_for(np.radians(yaw_deg)),
                                        observer=observer)
            self._sample(stage)
            self.stages.append({
                'stage': stage, 'attempt': attempt,
                'success': bool(result.success), 'reason': result.reason,
                'seconds': (self.sim.steps - start) * self.sim.dt,
                'distance_rad': float(result.distance),
                'position_error_m': getattr(result, 'position_error_m', None),
                'orientation_error_rad': getattr(result, 'orientation_error_rad', None),
            })
            if result.success or result.reason not in ('cartesian_pose_error',
                                                       'tracking_error'):
                return result
        return result

    def _fail(self, reason, stage):
        self.failure, self.failure_stage = reason, stage
        self.success = False
        return False


class TopGraspTask(SuccessorTask):
    """C：固定顶抓姿态，闭合双指后提升到指定高度并保持。"""

    defaults = {
        'approach_height': 0.13,
        'grasp_dz': 0.009,
        'lift_height': 0.08,
        'hold_seconds': 0.8,
        'open_width': 0.07,
        'close_width': 0.017,
        'grasp_force': FINGER_MAX_FORCE,
        'min_grip_force': 1.0,
        'min_lift': 0.06,
        'slip_tolerance_m': 0.01,
        'yaw_deg': 0.0,
        'tilt_tolerance_deg': 15.0,
        'rest_speed': 0.02,
    }

    failure_text = {
        'grasp_approach_blocked': '顶抓下降被工件挡住（开口/位姿不匹配）',
        'grasp_no_contact': '闭合后双指未同时接触工件',
        'hand_contact': '工件与手掌/臂本体接触',
        'object_not_lifted': '闭合后提升时工件未被带起',
        'object_slipped': '提升/保持中工件相对夹爪滑移',
        'object_tipped': '提升后工件倾角过大',
        'object_unstable': '保持阶段工件未稳定',
        'not_released': '任务结束时仍与工件接触',
    }

    def run(self):
        P = self.params
        obj = object_state(self.sim, self.block)
        x, y, z = obj['position']
        self.initial_position = obj['position'].copy()
        self.initial_tilt = float(obj['tilt_deg'])

        set_gripper(self.sim, P['open_width'], P['grasp_force'], 0.06)
        self._step_stage('open_gripper', 0.5)
        if not self._move('approach_above', (x, y, z + P['approach_height'])).success:
            return self._fail(self.stages[-1]['reason'], 'approach_above')
        if not self._move('descend', (x, y, z + P['grasp_dz'])).success:
            return self._fail('grasp_approach_blocked', 'descend')

        set_gripper(self.sim, P['close_width'], P['grasp_force'], 0.05)
        self._step_stage('close', 0.8)
        self.closed_widths = finger_widths(self.sim)
        f1, f2, active = finger_contact_forces(self.sim, self.block)
        self.grasp_contact = {'force_1': f1, 'force_2': f2,
                              'active_links': list(active),
                              'widths': list(self.closed_widths)}
        if min(f1, f2) < P['min_grip_force'] or len(active) < 2:
            return self._fail('grasp_no_contact', 'close')
        if non_finger_contacts(self.sim, self.block):
            return self._fail('hand_contact', 'close')
        self.grasp_local = relative_pose(self.sim, self.block)
        held_z = object_state(self.sim, self.block)['position'][2]

        lift_target = z + P['grasp_dz'] + P['lift_height']
        if not self._move('lift', (x, y, lift_target)).success:
            return self._fail(self.stages[-1]['reason'], 'lift')
        self._sample('lift')
        lifted = object_state(self.sim, self.block)
        self.lift_rise = float(lifted['position'][2] - held_z)
        self.slip = float(np.linalg.norm(
            relative_pose(self.sim, self.block)[0] - self.grasp_local[0]))
        if self.lift_rise < P['min_lift']:
            return self._fail('object_not_lifted', 'lift')
        if self.slip > P['slip_tolerance_m']:
            return self._fail('object_slipped', 'lift')
        if non_finger_contacts(self.sim, self.block):
            return self._fail('hand_contact', 'lift')

        hold_start_z = lifted['position'][2]
        start = self.sim.steps
        self._step_stage('hold', P['hold_seconds'])
        final = object_state(self.sim, self.block)
        self.hold_drop = float(hold_start_z - final['position'][2])
        self.hold_drift = float(np.linalg.norm(
            relative_pose(self.sim, self.block)[0] - self.grasp_local[0]))
        self.final_tilt = float(final['tilt_deg'])
        self.final_speed = float(final['linear_speed'])
        self.final_angular_speed = float(final['angular_speed'])
        self.hold_seconds = (self.sim.steps - start) * self.sim.dt
        checks = [
            ('object_not_lifted', self.lift_rise >= P['min_lift']),
            ('object_slipped', max(self.slip, self.hold_drift) <= P['slip_tolerance_m']),
            ('hand_contact', self.hand_contact_samples == 0),
            ('object_tipped', self.final_tilt <= P['tilt_tolerance_deg']),
            ('object_unstable', self.hold_drop <= 0.005
             and self.final_speed <= P['rest_speed']),
        ]
        failed = [name for name, ok in checks if not ok]
        self.success = not failed
        self.failure = failed[0] if failed else None
        self.failure_stage = 'hold' if failed else None
        return self.success

    def result(self):
        P = self.params
        final = object_state(self.sim, self.block)
        return {
            'success': bool(self.success),
            'failure': self.failure,
            'failure_stage': self.failure_stage,
            'failure_text': self.failure_text.get(self.failure, self.failure),
            'task': 'C_top_grasp',
            'initial_object_position': self.initial_position.tolist(),
            'final_object_position': final['position'].tolist(),
            'final_tilt_deg': float(getattr(self, 'final_tilt', final['tilt_deg'])),
            'closed_finger_widths': list(getattr(self, 'closed_widths', (None, None))),
            'grasp_contact': getattr(self, 'grasp_contact', {}),
            'lift_rise_m': float(getattr(self, 'lift_rise', float('nan'))),
            'slip_m': float(getattr(self, 'slip', float('nan'))),
            'hold_drop_m': float(getattr(self, 'hold_drop', float('nan'))),
            'hold_drift_m': float(getattr(self, 'hold_drift', float('nan'))),
            'hold_seconds': float(getattr(self, 'hold_seconds', 0.0)),
            'sim_seconds': float(self.now()),
            'stage_trace': self.stages,
        }


class PushTask(SuccessorTask):
    """B：闭合夹爪，用指外侧推工件局部 -y 面，直到工件触到挡板。"""

    defaults = {
        'push_height': PUSH_HEIGHT,
        'push_yaw_deg': 90.0,
        'gripper_width': 0.004,
        'approach_gap': 0.06,
        'retreat_gap': 0.09,
        'increment': 0.01,
        'min_displacement': 0.10,
        'min_face_contacts': 1,
        'tolerance_x': 0.02,
        'wall_tolerance': 0.01,
        'tilt_tolerance_deg': 12.0,
        'rest_speed': 0.01,
        'max_tilt_abort_deg': 25.0,
        'settle_seconds': 1.0,
    }

    failure_text = {
        'grasp_no_contact': '闭合后双指未同时接触工件',
        'hand_contact': '工件与手掌/臂本体接触',
        'no_face_contact': '未在指定局部侧面建立接触',
        'object_not_moved': '工件未被推动到目标区',
        'object_not_reached_wall': '工件未推到挡板',
        'object_tipped': '推动中/后工件倾角过大',
        'object_off_axis': '工件偏离推动轴线',
        'object_unstable': '静置后工件未停下',
        'push_path_failed': '推动路径执行失败',
    }

    @staticmethod
    def _yaw_extent(sim, body, half):
        _, quat = p.getBasePositionAndOrientation(body, physicsClientId=sim.client)
        m = np.asarray(p.getMatrixFromQuaternion(quat), dtype=float).reshape(3, 3)
        return float(sum(abs(m[1, i]) * half[i] for i in range(3)))

    def _yaw_deg(self):
        _, quat = p.getBasePositionAndOrientation(self.block,
                                                  physicsClientId=self.sim.client)
        m = np.asarray(p.getMatrixFromQuaternion(quat), dtype=float).reshape(3, 3)
        return float(np.degrees(np.arctan2(m[1, 0], m[0, 0])))

    def _finger_front_y(self):
        """推 +y 时最先接触工件的表面：两指包围盒的最大 y。"""
        front = []
        for link in (9, 10):
            aabb = p.getAABB(self.sim.robot, link, physicsClientId=self.sim.client)
            front.append(aabb[1][1])
        return float(max(front))

    def _wall_contact(self):
        return len(p.getContactPoints(bodyA=self.block, bodyB=self.wall,
                                      physicsClientId=self.sim.client)) > 0

    def run(self):
        P = self.params
        if self.wall is None:
            self.wall = create_wall(self.sim)
        obj = object_state(self.sim, self.block)
        self.initial_position = obj['position'].copy()
        self.initial_tilt = float(obj['tilt_deg'])
        self.half_y = self._yaw_extent(self.sim, self.block, HALF)
        self.initial_yaw = self._yaw_deg()
        x, y = float(obj['position'][0]), float(obj['position'][1])

        yaw = float(P['push_yaw_deg'])
        set_gripper(self.sim, P['gripper_width'], FINGER_MAX_FORCE, 0.06)
        self._step_stage('close_gripper', 0.8)
        self.closed_widths = finger_widths(self.sim)

        approach_y = y - self.half_y - P['approach_gap']
        if not self._move('approach', (x, approach_y, P['push_height']),
                          yaw_deg=yaw).success:
            return self._fail('push_path_failed', 'approach')
        self.finger_front_offset = float(self._finger_front_y() - self.sim.pose()[0][1])

        target = WALL_Y - 2.0 * self.half_y - self.finger_front_offset + 0.006
        self.push_target_y = float(target)
        ee_y = approach_y
        self.face_contacts = 0
        self.wall_seen = False
        while ee_y < target:
            ee_y = min(ee_y + P['increment'], target)
            if not self._move('push', (x, ee_y, P['push_height']), yaw_deg=yaw).success:
                return self._fail('push_path_failed', 'push')
            self._sample('push')
            self._count_face_contacts()
            if self._wall_contact():
                self.wall_seen = True
                break
            if object_state(self.sim, self.block)['tilt_deg'] > P['max_tilt_abort_deg']:
                return self._fail('object_tipped', 'push')
        if not self.wall_seen and self._wall_contact():
            self.wall_seen = True

        self._move('retreat', (x, ee_y - P['retreat_gap'], P['push_height']), yaw_deg=yaw)
        self._step_stage('settle', P['settle_seconds'])

        final = object_state(self.sim, self.block)
        self.final_position = final['position'].copy()
        self.final_tilt = float(final['tilt_deg'])
        self.final_speed = float(final['linear_speed'])
        self.displacement = float(final['position'][1] - self.initial_position[1])
        self.final_half_y = self._yaw_extent(self.sim, self.block, HALF)
        front_face = float(final['position'][1] + self.final_half_y)
        self.wall_gap = float(WALL_Y - front_face)
        self.x_error = float(abs(final['position'][0] - 0.45))
        self.release_contacts = len([c for c in contact_points(self.sim, self.block)
                                     if c['normal_force'] > 0.0])
        checks = [
            ('hand_contact', self.hand_contact_samples == 0),
            ('no_face_contact', self.face_contacts >= self.params['min_face_contacts']),
            ('object_not_moved', self.displacement >= self.params['min_displacement']),
            ('object_not_reached_wall', self.wall_seen and self.wall_gap <= self.params['wall_tolerance']),
            ('object_tipped', self.final_tilt <= self.params['tilt_tolerance_deg']),
            ('object_unstable', self.final_speed <= self.params['rest_speed']),
            ('object_off_axis', self.x_error <= self.params['tolerance_x']),
        ]
        failed = [name for name, ok in checks if not ok]
        self.success = not failed
        self.failure = failed[0] if failed else None
        self.failure_stage = 'settle' if failed else None
        return self.success

    def _count_face_contacts(self):
        for c in contact_points(self.sim, self.block):
            if c['normal_force'] <= 0.0 or c['link'] not in (9, 10):
                continue
            # 局部 -y 面：局部坐标下该面固定在 -HALF[1]，与世界朝向无关
            local = object_local_point(self.sim, self.block, c['position'])
            if abs(local[1] + HALF[1]) <= 0.012 and abs(local[0]) <= HALF[0] + 0.01:
                self.face_contacts += 1

    def result(self):
        final = object_state(self.sim, self.block)
        return {
            'success': bool(self.success),
            'failure': self.failure,
            'failure_stage': self.failure_stage,
            'failure_text': self.failure_text.get(self.failure, self.failure),
            'task': 'B_side_push',
            'initial_object_position': self.initial_position.tolist(),
            'final_object_position': final['position'].tolist(),
            'final_tilt_deg': float(getattr(self, 'final_tilt', final['tilt_deg'])),
            'closed_finger_widths': list(getattr(self, 'closed_widths', (None, None))),
            'push_target_y': float(getattr(self, 'push_target_y', float('nan'))),
            'finger_front_offset': float(getattr(self, 'finger_front_offset', float('nan'))),
            'displacement_m': float(getattr(self, 'displacement', float('nan'))),
            'wall_gap_m': float(getattr(self, 'wall_gap', float('nan'))),
            'wall_contact': bool(getattr(self, 'wall_seen', False)),
            'face_contact_samples': int(getattr(self, 'face_contacts', 0)),
            'x_error_m': float(getattr(self, 'x_error', float('nan'))),
            'initial_yaw_deg': float(getattr(self, 'initial_yaw', float('nan'))),
            'final_yaw_deg': float(self._yaw_deg()),
            'yaw_drift_deg': float(abs((self._yaw_deg() - getattr(self, 'initial_yaw', 0.0)
                                        + 180.0) % 360.0 - 180.0)),
            'rest_speed': float(getattr(self, 'final_speed', float('nan'))),
            'release_contacts': int(getattr(self, 'release_contacts', -1)),
            'sim_seconds': float(self.now()),
            'stage_trace': self.stages,
        }
