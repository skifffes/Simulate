"""把专家执行过程录成 20 Hz 的 LeRobot 观测-动作序列。

与实验 1 的 VLAExpertTask 语义一致：
    在每个控制周期开始时记录当前观测与“即将执行”的动作（7 关节目标 + 夹爪目标），
    然后保持该命令 12 个物理步（240 Hz -> 20 Hz）。

区别是这里不依赖 GraspPlaceTask，而是包住实验 2 的 SkillExpert，
因此 A0/A90/B/C 都能用同一套录制逻辑。
"""
from __future__ import annotations

import numpy as np

from manipulation import FINGER_MAX_WIDTH, finger_widths
from robot_controller import MotionResult

from .experts import SkillExpert
from .scene2 import grasp_quat


class RecordingExpert(SkillExpert):
    def __init__(self, sim, block, camera, steps_per_action: int = 12):
        super().__init__(sim, block)
        self.camera = camera
        self.spa = int(steps_per_action)
        self.frames = {name: [] for name in camera.cameras}
        self.states = []
        self.actions = []
        self._cmd = np.asarray(sim.q(), dtype=float).copy()
        self._grip_cmd = float(np.mean(finger_widths(sim)))

    # ---------- 录制原语 ----------
    def _record(self):
        frames = self.camera.capture()
        for name in self.frames:
            self.frames[name].append(np.asarray(frames[name], dtype=np.uint8).copy())
        self.states.append(np.concatenate([self.sim.q(),
                                           np.asarray(finger_widths(self.sim))]))
        self.actions.append(np.concatenate([self._cmd, [self._grip_cmd]]))

    def _hold(self, ticks: int = 1):
        for _ in range(ticks):
            self._record()
            for _ in range(self.spa):
                self.sim.command(self._cmd)
                self.sim.step()
                q = self.sim.q()
                if np.any(q < self.sim.lower - 0.015) or np.any(q > self.sim.upper + 0.015):
                    return 'joint_limit_during_motion'
                if self.sim.collisions():
                    return 'collision_during_motion'
        return None

    def _grip(self, width: float):
        super()._grip(width)
        self._grip_cmd = float(np.clip(width, 0.0, FINGER_MAX_WIDTH))

    # ---------- 覆盖专家原语 ----------
    def _settle(self, seconds: float):
        ticks = max(1, int(round(seconds / (self.sim.dt * self.spa))))
        self._hold(ticks)

    def _move_q(self, q, tag, tol: float = 0.06):
        q = np.asarray(q, dtype=float)
        begin = self.sim.q().copy()
        delta = q - begin
        n_steps = max(1, int(np.ceil(float(np.linalg.norm(delta)) / 0.004)))
        n_ticks = max(1, int(np.ceil(n_steps / self.spa)))
        for k in range(1, n_ticks + 1):
            self._cmd = begin + delta * min(k * self.spa / n_steps, 1.0)
            reason = self._hold(1)
            if reason is not None:
                self.trace.append({'tag': tag, 'ok': False, 'reason': reason})
                return MotionResult(False, reason, n_ticks * self.spa * self.sim.dt, 0.0,
                                    True, False, reason == 'collision_during_motion',
                                    reason == 'joint_limit_during_motion', None)
        error = float(np.linalg.norm(self.sim.q() - q))
        reached = error < tol
        result = MotionResult(reached, 'ok' if reached else 'tracking_error',
                              n_ticks * self.spa * self.sim.dt, 0.0, True, reached, False,
                              False, error)
        self.trace.append({'tag': tag, 'ok': bool(result.success), 'reason': result.reason,
                           'err': round(error, 4)})
        return result

    def _move(self, pos, yaw, tag):
        q, ok, _ = self.ctl.solve(np.asarray(pos, dtype=float), grasp_quat(yaw))
        if not ok:
            self.trace.append({'tag': tag, 'ok': False, 'reason': 'ik_joint_limits'})
            return MotionResult(False, 'ik_joint_limits', 0, 0, False, False, False, False, None)
        # 推动时物体阻力会让关节跟踪误差变大，用更宽的阈值；最终是否推到位由物体位置判定。
        tol = 0.25 if tag.startswith('b_push') or tag == 'b_contact' else 0.06
        return self._move_q(q, tag, tol=tol)

    def _spin(self, p0, p1, yaw0, yaw1, n=10, tag='spin'):
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        ways, prev = [], self.sim.q()
        for i in range(1, n + 1):
            a = i / n
            q, ok, _ = self.ctl.solve(p0 + (p1 - p0) * a,
                                      grasp_quat(yaw0 + (yaw1 - yaw0) * a), rest=prev)
            if not ok:
                return False, 'ik_joint_limits'
            ways.append(q)
            prev = q
        for q in ways:
            result = self._move_q(q, tag)
            if not result.success:
                return False, result.reason
        return True, 'ok'

    # ---------- 导出 ----------
    def episode_arrays(self):
        frames = {name: np.stack(values) for name, values in self.frames.items()}
        return frames, np.asarray(self.states, dtype=np.float32), \
            np.asarray(self.actions, dtype=np.float32)
