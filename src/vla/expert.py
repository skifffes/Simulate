"""专家轨迹采集：把 0C 的抓放流程重排为 20 Hz 的“观测 -> 动作”序列。

与实验 0C 的唯一区别是执行方式：专家在这里把关节目标在 20 Hz 控制周期上
离散化，每个动作在 12 个仿真步内保持不变，与后面 VLA 部署时的执行方式完全
一致。IK、关节插值步长、夹爪力/速度、成功判定全部复用原有实现。
"""
import numpy as np

from grasp_task import DEFAULT_PARAMS, FAIL_TEXT, GraspPlaceTask
from manipulation import (
    FINGER_MAX_WIDTH, finger_contact_forces, finger_widths, non_finger_contacts,
    object_state, relative_pose, set_gripper,
)
from robot_controller import MotionResult

from .config import STEPS_PER_ACTION


class VLAExpertTask(GraspPlaceTask):
    """一次可录制 episode。frames/states/actions 与 20 Hz 控制周期严格对齐。"""

    def __init__(self, sim, controller, block, params=None, camera=None,
                 steps_per_action=STEPS_PER_ACTION, joint_step=0.004, settle_steps=60):
        super().__init__(sim, controller, block, params, on_step=None,
                         record_interval=4, contact_interval=8)
        if camera is None:
            raise ValueError('camera rig is required')
        self.camera = camera
        self.steps_per_action = max(1, int(steps_per_action))
        self.joint_step = float(joint_step)
        self.settle_steps = int(settle_steps)
        self.frames = []
        self.states = []
        self.actions = []
        self.stage_labels = []
        self._setpoint = sim.q().copy()
        self.gripper_action = float(np.mean(finger_widths(sim)))

    # ---------- 控制周期原语 ----------
    def _gripper(self, width, force, speed):
        set_gripper(self.sim, width, force, speed)
        self.gripper_action = float(np.clip(width, 0.0, FINGER_MAX_WIDTH))

    def _record(self, stage):
        frames = self.camera.capture()
        self.frames.append({name: frame for name, frame in frames.items()})
        self.states.append(np.concatenate([self.sim.q(), np.asarray(finger_widths(self.sim))]))
        self.actions.append(np.concatenate([self._setpoint, [self.gripper_action]]))
        self.stage_labels.append(stage)

    def _tick(self, stage, check_motion=False):
        """一个 20 Hz 控制周期：先记录当前观测与即将执行的命令，再物理推进。"""
        self._record(stage)
        sim = self.sim
        for _ in range(self.steps_per_action):
            sim.step()
            self._sample_due(stage)
            if check_motion:
                q = sim.q()
                if bool(np.any(q < sim.lower - 0.015) or np.any(q > sim.upper + 0.015)):
                    return 'joint_limit_during_motion'
                if sim.collisions():
                    return 'collision_during_motion'
        return None

    def _step_stage(self, stage, seconds):
        start = self.sim.steps
        steps = max(1, int(round(seconds / self.sim.dt)))
        ticks = max(1, int(np.ceil(steps / self.steps_per_action)))
        self.sim.command(self._setpoint)
        for _ in range(ticks):
            reason = self._tick(stage)
            if reason is not None:
                self.failure, self.failure_stage = reason, stage
                self.stages.append({'stage': stage, 'success': False, 'reason': reason,
                                    'seconds': (self.sim.steps - start) * self.sim.dt})
                return False
        self._sample(stage)
        self.stages.append({'stage': stage, 'success': True, 'reason': 'ok',
                            'seconds': (self.sim.steps - start) * self.sim.dt})
        return True

    def _move(self, stage, target):
        sim = self.sim
        start = sim.steps
        goal = np.asarray(target, dtype=float)
        solution, ok, _ = self.ctl.solve(goal, self.params['orientation'])
        if not ok:
            self.stages.append({'stage': stage, 'success': False, 'reason': 'ik_joint_limits',
                                'seconds': 0.0})
            return MotionResult(False, 'ik_joint_limits', 0, 0, False, False, False, False, None)

        begin = sim.q().copy()
        delta = solution - begin
        distance = float(np.linalg.norm(delta))
        n_steps = max(1, int(np.ceil(distance / self.joint_step)))
        n_ticks = max(1, int(np.ceil(n_steps / self.steps_per_action)))
        settle_ticks = int(np.ceil(self.settle_steps / self.steps_per_action))
        travelled = 0.0
        previous = begin.copy()
        tick_count = 0
        for k in range(1, n_ticks + 1):
            progress = min(k * self.steps_per_action / n_steps, 1.0)
            self._setpoint = begin + delta * progress
            sim.command(self._setpoint)
            reason = self._tick(stage, check_motion=True)
            tick_count += 1
            q = sim.q()
            travelled += float(np.linalg.norm(q - previous))
            previous = q.copy()
            if reason is not None:
                self.stages.append({'stage': stage, 'success': False, 'reason': reason,
                                    'seconds': (sim.steps - start) * sim.dt})
                return MotionResult(False, reason, (sim.steps - start) * sim.dt, travelled,
                                    True, False, reason == 'collision_during_motion',
                                    reason == 'joint_limit_during_motion', None)
        for _ in range(settle_ticks):
            sim.command(self._setpoint)
            reason = self._tick(stage, check_motion=True)
            tick_count += 1
            if reason is not None:
                self.stages.append({'stage': stage, 'success': False, 'reason': reason,
                                    'seconds': (sim.steps - start) * sim.dt})
                return MotionResult(False, reason, (sim.steps - start) * sim.dt, travelled,
                                    True, False, reason == 'collision_during_motion',
                                    reason == 'joint_limit_during_motion', None)

        self._sample(stage)
        error = float(np.linalg.norm(sim.q() - solution))
        reached = error < 0.06
        actual, quat = sim.pose()
        position_error = float(np.linalg.norm(actual - goal))
        orientation_error = float(2 * np.arccos(np.clip(abs(np.dot(quat, self.params['orientation'])), 0, 1)))
        success = bool(reached and position_error < 0.025 and orientation_error < 0.15)
        reason = 'ok'
        if not success:
            reason = 'tracking_error' if not reached else 'cartesian_pose_error'
        self.stages.append({
            'stage': stage, 'success': success, 'reason': reason,
            'seconds': (sim.steps - start) * sim.dt, 'distance_rad': travelled,
            'position_error_m': position_error, 'control_ticks': tick_count,
        })
        return MotionResult(success, reason, (sim.steps - start) * sim.dt, travelled,
                            True, reached, False, False, error)

    # ---------- 主流程（与 0C 完全一致的阶段与判定） ----------
    def run(self):
        P = self.params
        self._gripper(P['open_width'], P['open_force'], P['open_speed'])
        self._step_stage('open_gripper', 0.5)

        for stage, target in (('approach_pre', P['pre_grasp']), ('approach', P['grasp'])):
            if stage == 'approach':
                self._gripper(P['preclose_width'], P['grasp_force'], P['close_speed'])
                if not self._step_stage('preclose', P['preclose_seconds']):
                    return False
            result = self._move(stage, target)
            if not result.success:
                return self._fail(result.reason, stage)

        self._gripper(P['close_width'], P['grasp_force'], P['close_speed'])
        if not self._step_stage('close', P['close_seconds']):
            return False
        self.closed_widths = finger_widths(self.sim)
        f1, f2, active = finger_contact_forces(self.sim, self.block)
        self.grasp_contact_record = {
            'force_1': f1, 'force_2': f2, 'active_links': list(active),
            'widths': list(self.closed_widths),
        }
        if len(active) < 2:
            return self._fail('grasp_no_contact', 'close')
        if non_finger_contacts(self.sim, self.block):
            return self._fail('hand_contact', 'close')
        self.grasp_local = relative_pose(self.sim, self.block)
        grasped_z = object_state(self.sim, self.block)['position'][2]

        result = self._move('lift', P['lift'])
        if not result.success:
            return self._fail(result.reason, 'lift')
        lifted = object_state(self.sim, self.block)
        self.lift_height = float(lifted['position'][2] - grasped_z)
        local = relative_pose(self.sim, self.block)
        self.slip = float(np.linalg.norm(local[0] - self.grasp_local[0]))
        if self.lift_height < P['lift_height_target']:
            return self._fail('object_not_lifted', 'lift')
        if self.slip > P['slip_tolerance_m']:
            return self._fail('object_slipped', 'lift')
        if non_finger_contacts(self.sim, self.block):
            return self._fail('hand_contact', 'lift')

        for stage, target in (('transport', P['place_above']), ('lower', P['place'])):
            result = self._move(stage, target)
            if not result.success:
                return self._fail(result.reason, stage)
            local = relative_pose(self.sim, self.block)
            self.slip = max(self.slip, float(np.linalg.norm(local[0] - self.grasp_local[0])))
            if self.slip > P['slip_tolerance_m']:
                return self._fail('object_slipped', stage)
            if non_finger_contacts(self.sim, self.block):
                return self._fail('hand_contact', stage)

        self._gripper(P['open_width'], P['open_force'], P['open_speed'])
        if not self._step_stage('release', P['release_seconds']):
            return False
        result = self._move('retreat', P['retreat'])
        if not result.success:
            return self._fail(result.reason, 'retreat')

        if not self._step_stage('settle', P['settle_seconds']):
            return False
        self._finalize()
        return self.success

    def transition_count(self):
        return len(self.actions)

    def episode_arrays(self):
        return (self.frames, np.asarray(self.states, dtype=np.float32),
                np.asarray(self.actions, dtype=np.float32), list(self.stage_labels))

    def failure_text(self):
        return FAIL_TEXT.get(self.failure, self.failure)


__all__ = ['VLAExpertTask', 'DEFAULT_PARAMS']
