"""15 Hz（每控制周期 16 个物理步）可行性检查 —— 为 4.2 的时间基准决策提供证据。

做法：同一场景、同一专家，只把 `steps_per_action` 从 12 改成 16（240/15），
用**同一个严格评价器**判定成功与否，并与 20 Hz（12 步）的 pilot 结果对照。

结论只用于"15 Hz 是否可行"，不改变本次 4.1 数据（数据仍为 20 Hz）。

用法：
    python scripts/exp41/fps_feasibility.py --out results/experiment_4_1/dataset
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
from exp41.experts41 import A90GoalExpert, B90GoalExpert  # noqa: E402
from exp41.strict_eval import (  # noqa: E402
    DEFAULT_CONFIG, PhysicsMonitor, evaluate_A, evaluate_B, evaluate_initial_B,
)


class _NullCamera:
    cameras = {}

    def capture(self):
        return {}


def run(skill: str, *, spa: int, seed: int) -> dict:
    if skill == 'A90':
        goal = scene41.A_GOALS[4]
        scene = scene41.build_a_scene(goal, seed=seed)
        obj = robot = None
    else:
        obj_id, goal_id = scene41.B_ALIGNED_PAIRS[1]
        goal = next(g for g in scene41.B_GOALS if g.goal_id == goal_id)
        obj = next(o for o in scene41.B_OBJECT_STARTS if o['start_id'] == obj_id)
        robot = scene41.B_ROBOT_STARTS[0]
        scene = scene41.build_b_scene(obj, robot, seed=seed)
    mon = PhysicsMonitor(scene, wall_body=scene.wall).attach()
    start = scene.object_state()
    initial = evaluate_initial_B(scene, DEFAULT_CONFIG) if skill == 'B90' else None
    try:
        ex = (A90GoalExpert if skill == 'A90' else B90GoalExpert)(
            scene.sim, scene.block, _NullCamera(), spa)
        ex.run_goal(goal)
        ex._settle(0.8)
        if skill == 'A90':
            ev = evaluate_A(scene, mon, goal, DEFAULT_CONFIG, start_obj_pos=start['pos'])
        else:
            ev = evaluate_B(scene, mon, goal, DEFAULT_CONFIG, start_obj_pos=start['pos'],
                            initial={**initial, 'yaw': start['yaw']})
        return {'spa': spa, 'hz': 240 / spa, 'strict_success': ev['strict_success'],
                'failure_reason': ev['failure_reason'],
                'metrics': {k: ev['metrics'].get(k) for k in
                            ('position_error_m', 'yaw_error_deg', 'advance_x_m', 'terminal_gap_m',
                             'wall_contact_steps', 'tilt_deg')}}
    finally:
        mon.detach()
        scene.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / 'results' / 'experiment_4_1' / 'dataset'))
    ap.add_argument('--seeds', type=int, default=3)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {'A90': {}, 'B90': {}}
    for skill in ('A90', 'B90'):
        for spa in (12, 16):
            runs = [run(skill, spa=spa, seed=410000 + i) for i in range(args.seeds)]
            res[skill][f'{240 // spa}Hz'] = {
                'steps_per_action': spa, 'hz': 240 / spa,
                'n': len(runs), 'n_success': sum(1 for r in runs if r['strict_success']),
                'runs': runs,
            }
    res['note'] = ('12 步 = 20 Hz（本次数据使用的基准）；16 步 = 15 Hz（DROID 原始控制频率）。'
                   '两组都用同一个严格评价器；成功率相同说明 15 Hz 在该专家上同样可行，'
                   '但动作与状态的**时间关系**不同（一步 62.5 ms vs 50 ms），'
                   '跨频率训练/推理必须做时间归一化。')
    (out / 'fps_feasibility.json').write_text(json.dumps(res, ensure_ascii=False, indent=2,
                                                         default=float), encoding='utf-8')
    for skill in ('A90', 'B90'):
        print(skill, {k: (v['n_success'], v['n']) for k, v in res[skill].items()})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
