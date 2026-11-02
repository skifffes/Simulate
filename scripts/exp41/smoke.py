"""实验 4.1 冒烟测试：跑 1 条 A90 + 1 条 B90，打印严格评价结果与动作相位检查。

用法（服务器，simulate_vla venv）：
    python scripts/exp41/smoke.py [--skill A90|B90|both]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / 'src', ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from exp41 import scene41  # noqa: E402
from exp41.run41 import run_attempt  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--skill', default='both', choices=['A90', 'B90', 'both'])
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--no-frames', action='store_true')
    args = ap.parse_args(argv)

    out = {}
    if args.skill in ('A90', 'both'):
        goal = scene41.A_GOALS[4]      # 中心目标
        att = run_attempt('A90', seed=args.seed, goal=goal, record_frames=not args.no_frames)
        ev = att.record['evaluation']
        out['A90'] = {'goal': goal.goal_id, 'expert_ok': att.record['expert_ok'],
                      'expert_reason': att.record['expert_reason'],
                      'strict_success': ev['strict_success'], 'failure_reason': ev['failure_reason'],
                      'checks': ev['checks'],
                      'metrics': {k: ev['metrics'][k] for k in
                                  ('position_error_m', 'yaw_error_deg', 'tilt_deg', 'lift_m',
                                   'transport_m', 'final_speed_m_s', 'ee_object_distance_m')},
                      'action_phase': att.record.get('action_phase'),
                      'n_steps': att.record['n_control_steps']}
    if args.skill in ('B90', 'both'):
        goal = scene41.B_GOALS[1]      # 中间墙面区域
        start = scene41.B_OBJECT_STARTS[1]
        robot = scene41.B_ROBOT_STARTS[0]
        att = run_attempt('B90', seed=args.seed, goal=goal, obj_start=start,
                          robot_start=robot, record_frames=not args.no_frames)
        ev = att.record['evaluation']
        out['B90'] = {'goal': goal.goal_id, 'obj_start': start['start_id'],
                      'robot_start': robot['robot_start_id'],
                      'expert_ok': att.record['expert_ok'],
                      'expert_reason': att.record['expert_reason'],
                      'strict_success': ev['strict_success'], 'failure_reason': ev['failure_reason'],
                      'checks': ev['checks'],
                      'metrics': {k: ev['metrics'][k] for k in
                                  ('advance_x_m', 'lateral_drift_m', 'terminal_gap_m',
                                   'wall_contact_steps', 'wall_contact_force_max_n',
                                   'wall_normal_abs_x_min', 'robot_block_contact_steps',
                                   'final_y', 'tilt_deg', 'final_speed_m_s')},
                      'action_phase': att.record.get('action_phase'),
                      'n_steps': att.record['n_control_steps'],
                      'initial': att.record.get('initial')}
    print(json.dumps(out, ensure_ascii=False, indent=2, default=float))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
