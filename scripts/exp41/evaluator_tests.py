"""实验 4.1 严格评价器的物理正负案例（Gate 2）。

每个案例都是一个**真实 PyBullet 场景 + 真实专家动作**：
  * 正例：A90 / B90 专家在合法条件下完成任务 → `strict_success=True`；
  * 负例：通过改变场景或专家参数构造"看起来像成功但物理上不满足"的情形，
    要求评价器给出**具体失败原因**，而不是笼统失败。

覆盖（prompt §7.4 / §11）：
  B90：真实推墙（正）、近墙不接触、初始已贴墙、初始穿透、推错方向、
       推到墙后回拖离开、机器人自己碰墙而方块没碰、目标区域不符、超高推进（顶翻）、
       过度推进（压墙）。
  A90：正常抓放（正）、未抬起、夹住后滑落、放错位置、姿态错误（长轴 Y）、未释放、
       未撤离、放置后不稳定（高处掉落）。

用法：
    python scripts/exp41/evaluator_tests.py [--out results/experiment_4_1/evaluation]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / 'src', ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from exp41 import scene41  # noqa: E402
from exp41.experts41 import A90GoalExpert, B90GoalExpert  # noqa: E402
from exp41.strict_eval import (  # noqa: E402
    DEFAULT_CONFIG, PhysicsMonitor, evaluate_A, evaluate_B, evaluate_initial_B,
)

STEPS_PER_ACTION = 12
RESULTS = []


def _ok(skill: str, name: str, ev: dict, expect: dict) -> dict:
    if expect.get('expect_success'):
        passed = bool(ev['strict_success'])
        detail = {'strict_success': ev['strict_success'], 'failure_reason': ev['failure_reason']}
    else:
        reason_in = expect.get('failure_reason_in')
        passed = (not ev['strict_success']) and (reason_in is None or ev['failure_reason'] in reason_in)
        detail = {'strict_success': ev['strict_success'], 'failure_reason': ev['failure_reason'],
                  'expected_any_of': reason_in}
    row = {'skill': skill, 'case': name, 'pass': bool(passed),
           'expected': expect, 'detail': detail,
           'checks': ev['checks'],
           'key_metrics': {k: v for k, v in ev['metrics'].items()
                           if k in ('position_error_m', 'yaw_error_deg', 'tilt_deg', 'lift_m',
                                    'transport_m', 'advance_x_m', 'terminal_gap_m',
                                    'wall_contact_steps', 'wall_contact_force_max_n',
                                    'robot_block_contact_steps', 'robot_wall_contact_steps',
                                    'final_y', 'final_speed_m_s', 'rebound_m')}}
    RESULTS.append(row)
    print(('PASS  ' if passed else 'FAIL  ') + f'[{skill}] {name}'
          + f"  -> {ev['failure_reason']}" if not passed else
          ('PASS  ' + f'[{skill}] {name}  -> success={ev["strict_success"]} ({ev["failure_reason"]})'))
    return row


def _run_B(goal, obj_start, robot_start, *, seed=1, **kw):
    """跑一次 B90（可选参数改变专家行为），返回 (ev, monitor_summary)。"""
    scene = scene41.build_b_scene(obj_start, robot_start, seed=seed)
    mon = PhysicsMonitor(scene, wall_body=scene.wall).attach()
    start = scene.object_state()
    initial = evaluate_initial_B(scene, DEFAULT_CONFIG)
    try:
        class _C:
            cameras = {}

            def capture(self):
                return {}

        ex = B90GoalExpert(scene.sim, scene.block, _C(), STEPS_PER_ACTION)
        ex.run_goal(goal, **kw)
        ex._settle(0.8)
        ev = evaluate_B(scene, mon, goal, DEFAULT_CONFIG, start_obj_pos=start['pos'],
                        initial={**initial, 'yaw': start['yaw']})
        return ev, mon.summary()
    finally:
        mon.detach()
        scene.close()


def _run_A(goal, *, seed=1, obj_pos=None, obj_yaw=0.0, **kw):
    scene = scene41.build_a_scene(goal, seed=seed)
    if obj_pos is not None:
        import pybullet as p

        p.resetBasePositionAndOrientation(scene.block, list(obj_pos), [0, 0, 0, 1],
                                          physicsClientId=scene.sim.client)
        for _ in range(60):
            scene.sim.step()
    mon = PhysicsMonitor(scene, wall_body=scene.wall).attach()
    start = scene.object_state()
    try:
        class _C:
            cameras = {}

            def capture(self):
                return {}

        ex = A90GoalExpert(scene.sim, scene.block, _C(), STEPS_PER_ACTION)
        ex.run_goal(goal, **kw)
        ex._settle(0.8)
        ev = evaluate_A(scene, mon, goal, DEFAULT_CONFIG, start_obj_pos=start['pos'])
        return ev, mon.summary()
    finally:
        mon.detach()
        scene.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / 'results' / 'experiment_4_1' / 'evaluation'))
    args = ap.parse_args(argv)
    out = Path(args.out)
    (out / 'calibration_cases').mkdir(parents=True, exist_ok=True)

    a_goal = scene41.A_GOALS[4]
    b_goal = scene41.B_GOALS[1]
    b_obj = scene41.B_OBJECT_STARTS[1]
    b_robot = scene41.B_ROBOT_STARTS[0]

    # ---------------- B90 正例 ----------------
    ev, mon = _run_B(b_goal, b_obj, b_robot)
    _ok('B90', 'positive_real_push_and_wall_contact', ev, {'expect_success': True})
    (out / 'calibration_cases' / 'B90_positive_contact.json').write_text(
        json.dumps({'eval': ev, 'contacts': mon}, ensure_ascii=False, indent=2, default=float),
        encoding='utf-8')

    # ---------------- B90 负例 ----------------
    ev, _ = _run_B(b_goal, b_obj, b_robot, force_wall_contact=False)
    _ok('B90', 'near_wall_without_contact', ev,
        {'expect_success': False, 'failure_reason_in': ['wall_contact_ever', 'valid_wall_contact']})

    ev, _ = _run_B(b_goal, b_obj, b_robot, direction=-1, force_wall_contact=False)
    _ok('B90', 'pushed_wrong_direction', ev,
        {'expect_success': False, 'failure_reason_in': ['advance_ok', 'wall_contact_ever',
                                                        'terminal_gap_ok']})

    ev, _ = _run_B(b_goal, b_obj, b_robot, push_z=0.05, force_wall_contact=False)
    _ok('B90', 'pushed_too_high_climbs_object', ev,
        {'expect_success': False, 'failure_reason_in': ['tilt_ok', 'terminal_gap_ok',
                                                        'wall_contact_ever', 'advance_ok']})

    ev, _ = _run_B(b_goal, b_obj, b_robot, force_wall_contact=True, overshoot=0.02)
    _ok('B90', 'overshoot_after_wall_contact', ev, {'expect_success': True})

    # 初始已贴墙 / 初始穿透：直接改初始物体位置
    ev, _ = _run_B(b_goal, {**b_obj, 'x': 0.75}, b_robot, force_wall_contact=False)
    _ok('B90', 'initial_already_touching_wall', ev,
        {'expect_success': False, 'failure_reason_in': ['initial_valid']})

    ev, _ = _run_B(b_goal, {**b_obj, 'x': 0.79}, b_robot, force_wall_contact=False)
    _ok('B90', 'initial_penetrates_wall', ev,
        {'expect_success': False, 'failure_reason_in': ['initial_valid']})

    # 目标区域不符：推到位但用错的目标去评价
    wrong_goal = scene41.B_GOALS[2]
    ev, _ = _run_B(wrong_goal, b_obj, b_robot)
    _ok('B90', 'wrong_goal_region', ev,
        {'expect_success': False, 'failure_reason_in': ['goal_region_ok', 'lateral_ok',
                                                        'terminal_gap_ok']})

    # 机器人自己碰墙而方块没碰：把 EE 直接开到墙边但不接触方块
    scene = scene41.build_b_scene({**b_obj, 'x': 0.30}, b_robot, seed=3)
    mon = PhysicsMonitor(scene, wall_body=scene.wall).attach()
    start = scene.object_state()
    initial = evaluate_initial_B(scene, DEFAULT_CONFIG)
    try:
        class _C:
            cameras = {}

            def capture(self):
                return {}

        ex = B90GoalExpert(scene.sim, scene.block, _C(), STEPS_PER_ACTION)
        ex._grip(0.014)
        ex._move((0.70, 0.21, 0.10), 0.0, 'robot_to_wall')
        ex._move((0.76, 0.21, 0.10), 0.0, 'robot_touch_wall')
        ex._settle(0.6)
        ev = evaluate_B(scene, mon, b_goal, DEFAULT_CONFIG, start_obj_pos=start['pos'],
                        initial={**initial, 'yaw': start['yaw']})
    finally:
        mon.detach()
        scene.close()
    _ok('B90', 'robot_touches_wall_block_does_not', ev,
        {'expect_success': False,
         'failure_reason_in': ['pushed_by_robot', 'advance_ok', 'wall_contact_ever']})

    # ---------------- A90 正例 ----------------
    ev, mon = _run_A(a_goal)
    _ok('A90', 'positive_grasp_place_release', ev, {'expect_success': True})
    (out / 'calibration_cases' / 'A90_positive.json').write_text(
        json.dumps({'eval': ev, 'contacts': mon}, ensure_ascii=False, indent=2, default=float),
        encoding='utf-8')

    # ---------------- A90 负例 ----------------
    ev, _ = _run_A(a_goal, skip_grasp=True)
    _ok('A90', 'no_grasp', ev,
        {'expect_success': False, 'failure_reason_in': ['grasped_by_two_fingers', 'lifted']})

    ev, _ = _run_A(a_goal, place_offset=(0.06, 0.03))
    _ok('A90', 'placed_at_wrong_position', ev,
        {'expect_success': False, 'failure_reason_in': ['position_ok']})

    ev, _ = _run_A(a_goal, force_place_yaw=np.pi / 2)
    _ok('A90', 'wrong_place_orientation', ev,
        {'expect_success': False, 'failure_reason_in': ['yaw_ok']})

    ev, _ = _run_A(a_goal, release=False, retreat=False)
    _ok('A90', 'not_released', ev,
        {'expect_success': False, 'failure_reason_in': ['released']})

    ev, _ = _run_A(a_goal, retreat=False)
    _ok('A90', 'no_retreat', ev,
        {'expect_success': False, 'failure_reason_in': ['retreated', 'released']})

    ev, _ = _run_A(a_goal, drop_from=0.05)
    _ok('A90', 'dropped_from_height_unstable', ev,
        {'expect_success': False,
         'failure_reason_in': ['position_ok', 'yaw_ok', 'stable', 'tilt_ok', 'no_penetration']})

    ev, _ = _run_A(a_goal, drop_mid=True)
    _ok('A90', 'slipped_mid_transport', ev,
        {'expect_success': False,
         'failure_reason_in': ['position_ok', 'yaw_ok', 'stable', 'tilt_ok', 'lifted',
                               'released', 'retreated', 'no_penetration']})

    total = len(RESULTS)
    passed = sum(1 for r in RESULTS if r['pass'])
    summary = {'eval_config': asdict(DEFAULT_CONFIG), 'n_cases': total, 'n_pass': passed,
               'all_pass': passed == total, 'cases': RESULTS}
    (out / 'evaluator_tests.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding='utf-8')
    print(f'\n{passed}/{total} cases passed')
    return 0 if passed == total else 1


if __name__ == '__main__':
    raise SystemExit(main())
