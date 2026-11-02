"""B90 目标条件可区分性测试（RQ5 的证据）。

问题：同一物体起点、不同墙面目标，当前 +X 专家能否靠"选择不同动作"完成？
做法：固定物体起点 `B_obj_y021` 与机器人起点 `B_robot_home`，对 3 个墙面目标各跑 2 个种子，
允许专家做横向修正；记录严格评价结果与专家失败原因。

预期（并如实记录）：离起点较近的目标成功，需要跨区横移的目标失败 →
说明**当前专家不能把"目标"当作独立控制维度**，主数据只用预配对 (起点, 目标)。

用法：python scripts/exp41/goal_conditioning_test.py --out results/experiment_4_1/dataset
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
    ap.add_argument('--out', default=str(ROOT / 'results' / 'experiment_4_1' / 'dataset'))
    ap.add_argument('--seeds', type=int, default=2)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    obj = next(o for o in scene41.B_OBJECT_STARTS if o['start_id'] == 'B_obj_y021')
    robot = scene41.B_ROBOT_STARTS[0]
    rows = []
    for goal in scene41.B_GOALS:
        for k in range(args.seeds):
            att = run_attempt('B90', seed=430000 + k, goal=goal, obj_start=obj,
                              robot_start=robot, record_frames=False)
            r = att.record
            rows.append({'goal_id': goal.goal_id, 'goal_y_range': [goal.y_lo, goal.y_hi],
                         'seed': r['seed'], 'strict_success': r['strict_success'],
                         'failure_reason': r['failure_reason'],
                         'expert_ok': r['expert_ok'], 'expert_reason': r['expert_reason'],
                         'final_y': r['metrics'].get('final_y'),
                         'advance_x_m': r['metrics'].get('advance_x_m'),
                         'wall_contact_steps': r['metrics'].get('wall_contact_steps'),
                         'lateral_needed': r['metrics'].get('lateral_needed_for_goal')})
            print(f"[{goal.goal_id}] seed={r['seed']} success={r['strict_success']} "
                  f"expert={r['expert_reason']} final_y={r['metrics'].get('final_y')}")
    summary = {
        'object_start': obj, 'robot_start': robot,
        'n': len(rows),
        'n_success': sum(1 for r in rows if r['strict_success']),
        'by_goal': {g.goal_id: {'attempts': sum(1 for r in rows if r['goal_id'] == g.goal_id),
                                'success': sum(1 for r in rows if r['goal_id'] == g.goal_id
                                               and r['strict_success'])}
                    for g in scene41.B_GOALS},
        'rows': rows,
        'conclusion': ('若只有与起点同一区域的目标成功、跨区目标失败，则说明当前 +X 专家'
                       '无法把目标作为独立控制维度，主数据只用预配对 (起点, 目标)。'),
    }
    (out / 'goal_conditioning_feasibility.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding='utf-8')
    print('success', summary['n_success'], '/', summary['n'], json.dumps(summary['by_goal'], ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
