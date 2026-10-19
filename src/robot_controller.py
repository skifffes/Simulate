"""逆运动学候选、关节插值执行、碰撞/限位与动作成本测量。"""
import numpy as np
import pybullet as p

class MotionResult:
    def __init__(self, success, reason, seconds, distance, ik, reached, collision, limit, error):
        self.success, self.reason = success, reason
        self.seconds, self.distance = seconds, distance
        self.ik, self.reached = ik, reached
        self.collision, self.limit, self.error = collision, limit, error

class RobotController:
    def __init__(self, sim, max_steps=1800):
        self.sim, self.max_steps = sim, max_steps

    def solve(self, position, orientation, rest=None):
        s = self.sim
        result = p.calculateInverseKinematics(s.robot, s.ee, position, orientation,
            lowerLimits=s.lower.tolist(), upperLimits=s.upper.tolist(),
            jointRanges=(s.upper-s.lower).tolist(), restPoses=(s.q() if rest is None else rest).tolist(),
            maxNumIterations=200, residualThreshold=1e-5, physicsClientId=s.client)
        q = np.asarray(result[:7])
        ok = bool(np.all(q >= s.lower - 1e-5) and np.all(q <= s.upper + 1e-5))
        return q, ok

    def move(self, target, sim_steps=None):
        s = self.sim
        target = np.asarray(target, dtype=float)
        if np.any(target < s.lower) or np.any(target > s.upper):
            return MotionResult(False, 'joint_limit_target', 0, 0, True, False, False, True, None)
        start = s.q().copy()
        delta = target - start
        distance = float(np.linalg.norm(delta))
        n = max(1, int(np.ceil(distance / 0.004)))
        limit = self.max_steps if sim_steps is None else sim_steps
        if n > limit:
            return MotionResult(False, 'path_step_limit', 0, distance, True, False, False, False, None)
        distance = 0.0
        previous = start
        for k in range(1, n + 61):
            self.sim.command(start + delta * min(k / n, 1.0))
            self.sim.step()
            q = self.sim.q()
            distance += float(np.linalg.norm(q - previous))
            previous = q
            seconds = k * self.sim.dt
            violation = bool(np.any(q < s.lower - 0.015) or np.any(q > s.upper + 0.015))
            if violation:
                return MotionResult(False, 'joint_limit_during_motion', seconds, distance, True, False, False, True, None)
            if s.collisions():
                return MotionResult(False, 'collision_during_motion', seconds, distance, True, False, True, False, None)
        err = float(np.linalg.norm(s.q() - target))
        reached = err < 0.06
        return MotionResult(reached, 'ok' if reached else 'tracking_error', seconds,
                            distance, True, reached, False, False, err)

    def move_pose(self, position, orientation, sim_steps=None):
        q, ik = self.solve(position, orientation)
        if not ik:
            return MotionResult(False, 'ik_joint_limits', 0, 0, False, False, False, False, None)
        result = self.move(q, sim_steps)
        actual, quat = self.sim.pose()
        result.position_error_m = float(np.linalg.norm(actual - np.asarray(position)))
        result.orientation_error_rad = float(2 * np.arccos(np.clip(abs(np.dot(quat, orientation)), 0, 1)))
        result.reached = bool(result.reached and result.position_error_m < 0.025 and result.orientation_error_rad < 0.15)
        result.success = bool(result.success and result.reached)
        if not result.success and result.reason == 'ok':
            result.reason = 'cartesian_pose_error'
        return result
