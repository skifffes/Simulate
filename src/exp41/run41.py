"""实验 4.1 运行器：一次原子技能尝试（场景 → 专家 → 稳定窗口 → 严格评价 → 记录）。

一次 attempt 的完整链路：
    Scene41(权威 body ID) → PhysicsMonitor(逐物理步接触) → Expert(真实控制)
    → settle 窗口（继续记录） → strict_eval → 记录 metrics/轨迹 → （若合格）写入 LeRobot

用法（被 pilot / collect 脚本调用；也可单独跑单条尝试）：
    python -m exp41.run41 --skill A90 --goal A_x00_y00 --seed 1 --no-write
"""
from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import dataclass, field

import numpy as np

from exp40 import camera_v2
from exp41 import scene41
from exp41.experts41 import A90GoalExpert, B90GoalExpert
from exp41.strict_eval import (
    DEFAULT_CONFIG, PhysicsMonitor, evaluate_A, evaluate_B, evaluate_initial_B,
)

ACTION_CONTRACT_VERSION = 'exp4.1-action-v1'
PHYSICS_HZ = 240
POLICY_HZ = 20                      # 与实验 2 一致；数据里另存 15 Hz 视图所需信息
STEPS_PER_ACTION = PHYSICS_HZ // POLICY_HZ   # 12

A90_PROMPT_TEMPLATE = ('pick up the orange block and place it in the target area at '
                       '({x:.3f}, {y:.3f}) with its long side aligned with X')
B90_PROMPT_TEMPLATE = ('push the orange block along the positive X direction to the wall '
                       'region y in [{y_lo:.2f}, {y_hi:.2f}]')


@dataclass
class Attempt:
    attempt_id: str
    skill: str
    seed: int
    goal_id: str
    scene_config: dict
    record: dict = field(default_factory=dict)
    frames: dict | None = None
    states: np.ndarray | None = None
    actions: np.ndarray | None = None


def _uuid(prefix: str, *parts) -> str:
    return f'{prefix}-' + uuid.uuid5(uuid.NAMESPACE_URL,
                                     '|'.join(str(p) for p in parts)).hex[:16]


def run_attempt(skill: str, *, seed: int, goal, obj_start: dict | None = None,
                robot_start: dict | None = None, allow_lateral: bool = True,
                record_frames: bool = True, camera_version: str = 'v3',
                cfg=DEFAULT_CONFIG, settle_seconds: float | None = None) -> Attempt:
    from vla.cameras import CameraRig

    attempt_id = _uuid('attempt', skill, goal.goal_id, seed,
                       (obj_start or {}).get('start_id'), (robot_start or {}).get('robot_start_id'))
    t0 = time.time()
    goal_point = None
    if skill == 'A90':
        scene = scene41.build_a_scene(goal, seed=seed, camera_version=camera_version)
        # 把目标区域落成具体目标点（确定性抖动），指令与评价都用该点
        gx, gy = scene41.a_goal_point(goal, seed)
        goal_point = (gx, gy)
        goal_eval = scene41.AGoal(goal.goal_id, gx, gy, goal.target_axis, goal.region_id)
        prompt = A90_PROMPT_TEMPLATE.format(x=gx, y=gy)
    elif skill == 'B90':
        scene = scene41.build_b_scene(obj_start, robot_start, seed=seed,
                                      camera_version=camera_version)
        prompt = B90_PROMPT_TEMPLATE.format(y_lo=goal.y_lo, y_hi=goal.y_hi)
        goal_eval = goal
    else:
        raise ValueError(skill)

    monitor = PhysicsMonitor(scene, wall_body=scene.wall).attach()
    attempt = Attempt(attempt_id=attempt_id, skill=skill, seed=seed, goal_id=goal.goal_id,
                      scene_config={'obj_start': obj_start, 'robot_start': robot_start,
                                    'goal': goal.as_dict() if hasattr(goal, 'as_dict') else None,
                                    'seed': seed})
    start_state = scene.object_state()
    initial_B = evaluate_initial_B(scene, cfg) if skill == 'B90' else None
    try:
        if record_frames:
            camera = CameraRig(scene.sim, scene.cameras, (256, 256), supersample=1)
            expert = (A90GoalExpert if skill == 'A90' else B90GoalExpert)(
                scene.sim, scene.block, camera, STEPS_PER_ACTION)
            result = expert.run_goal(goal_eval) if skill == 'A90' else \
                expert.run_goal(goal, allow_lateral=allow_lateral)
        else:
            # 不录帧时仍需要一个"有 cameras 属性"的占位相机，录制器才会初始化
            class _NullCamera:
                cameras = {}

                def capture(self):
                    return {}

            expert = (A90GoalExpert if skill == 'A90' else B90GoalExpert)(
                scene.sim, scene.block, _NullCamera(), STEPS_PER_ACTION)
            result = expert.run_goal(goal_eval) if skill == 'A90' else \
                expert.run_goal(goal, allow_lateral=allow_lateral)

        # ---- 独立稳定窗口（继续逐物理步记录）----
        settle_s = cfg.settle_seconds if settle_seconds is None else settle_seconds
        if record_frames:
            expert._settle(settle_s)
        else:
            for _ in range(int(settle_s / scene.sim.dt)):
                scene.sim.step()
        settle_pos = scene.object_state()['pos']
        settle_track_start = len(monitor.track)
        for _ in range(int(0.2 / scene.sim.dt)):
            scene.sim.step()
        settle_slice = np.asarray(monitor.track[settle_track_start:], dtype=float) \
            if len(monitor.track) > settle_track_start else None
        settle_track = None
        if settle_slice is not None and settle_slice.size:
            settle_track = {'x_max': float(settle_slice[:, 1].max()),
                            'x_min': float(settle_slice[:, 1].min())}

        if skill == 'A90':
            ev = evaluate_A(scene, monitor, goal_eval, cfg, start_obj_pos=start_state['pos'])
            record = {'kind': 'A90', 'expert_ok': bool(result['success']),
                      'expert_reason': result['reason'],
                      'expert_metrics': result.get('metrics', {}),
                      'goal': goal_eval.as_dict(), 'goal_point': list(goal_point),
                      'region_id': goal.goal_id, 'evaluation': ev}
        else:
            ev = evaluate_B(scene, monitor, goal, cfg, start_obj_pos=start_state['pos'],
                            initial={**initial_B,
                                     'yaw': start_state['yaw']}, settle_track=settle_track)
            record = {'kind': 'B90', 'expert_ok': bool(result['success']),
                      'expert_reason': result['reason'],
                      'expert_metrics': result.get('metrics', {}),
                      'goal': goal.as_dict(), 'initial': initial_B, 'evaluation': ev}

        # ---- 数据（仅原子技能，不含组合）----
        states = actions = None
        frames = None
        if record_frames:
            frames, states, actions = expert.episode_arrays()
            if len(actions) != len(states) or len(actions) < 5:
                record['data_error'] = f'length mismatch actions={len(actions)} states={len(states)}'

        # ---- 动作契约审计：action 是"下一控制周期的绝对关节目标" ----
        if actions is not None and states is not None and len(actions) > 2:
            # 验证 action[t] 与 state[t+1] 的关节差等于同一命令下的实际位移（相位检查）
            q_err = np.abs(actions[:-1, :7] - states[1:, :7]).max(axis=1)
            record['action_phase'] = {
                'max_abs_action_minus_next_state_rad': float(q_err.max()),
                'mean': float(q_err.mean()),
                'note': 'action[t] 是 t→t+1 的关节目标；若与 state[t+1] 接近则相位正确',
            }

        elapsed = time.time() - t0
        record.update({
            'attempt_id': attempt_id, 'skill': skill, 'seed': seed,
            'goal_id': goal.goal_id,
            'goal_point': list(goal_point) if goal_point else None,
            'camera_version': camera_v2.VERSION,
            'camera_sha256': camera_v2.CAMERA_CONFIG_SHA256,
            'action_contract_version': ACTION_CONTRACT_VERSION,
            'physics_hz': PHYSICS_HZ, 'policy_hz': POLICY_HZ,
            'steps_per_action': STEPS_PER_ACTION,
            'prompt': prompt,
            'strict_success': bool(ev['strict_success']),
            'failure_reason': ev['failure_reason'],
            'checks': ev['checks'],
            'metrics': ev['metrics'],
            'n_control_steps': int(len(actions)) if actions is not None else 0,
            'wall_seconds': round(elapsed, 2),
            'start_object': {'pos': start_state['pos'].tolist(),
                             'quat': start_state['quat'].tolist(), 'yaw': start_state['yaw']},
        })
        if initial_B is not None:
            record['initial_task_valid'] = initial_B['task_valid']
            record['initial_reasons'] = initial_B['reasons']
        attempt.record = record
        attempt.frames = frames
        attempt.states = states
        attempt.actions = actions
        return attempt
    finally:
        monitor.detach()
        scene.close()
