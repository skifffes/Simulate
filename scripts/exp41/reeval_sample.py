"""对已有 A90 目标做**重放抽样**，用 v1.1 评价器测量"双指/尾段角速度/释放后滑移"。

为什么需要重放：已存记录里 `contact_summary.robot_block.links` 当时错误地汇总了 `link_a`
（方块一侧，恒为 -1），因此无法从旧 JSON 判断左右指；而尾段轨迹没有保存，滑移也无法重算。
这里重放 6 条（每个目标区域抽样），用 v1.1 直接测量。

结论写入 `results/experiment_4_1/preflight/reeval_sample.json`；**不覆盖**任何旧标签。

用法：python scripts/exp41/reeval_sample.py --n 6 --out results/experiment_4_1/preflight
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
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--out', default=str(ROOT / 'results' / 'experiment_4_1' / 'preflight'))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, goal in enumerate(scene41.A_GOALS[:args.n]):
        att = run_attempt('A90', seed=460000 + i, goal=goal, record_frames=False)
        m = att.record['metrics']
        checks = att.record['checks']
        rows.append({
            'goal_id': goal.goal_id, 'seed': att.record['seed'],
            'eval_version': att.record['evaluation']['eval_version'],
            'strict_success': att.record['strict_success'],
            'failure_reason': att.record['failure_reason'],
            'finger9_contact_steps': m.get('finger9_contact_steps'),
            'finger9_max_force_n': m.get('finger9_max_force_n'),
            'finger10_contact_steps': m.get('finger10_contact_steps'),
            'finger10_max_force_n': m.get('finger10_max_force_n'),
            'slip_after_release_m': m.get('slip_after_release_m'),
            'ang_speed_tail_max_rad_s': m.get('ang_speed_tail_max_rad_s'),
            'two_finger_overlap_steps': m.get('two_finger_overlap_steps'),
            'two_finger_overlap_first_step': m.get('two_finger_overlap_first_step'),
            'two_finger_overlap_last_step': m.get('two_finger_overlap_last_step'),
            'final_ang_speed_rad_s': m.get('final_ang_speed_rad_s'),
            'checks': checks,
        })
        print(f"[{goal.goal_id}] success={att.record['strict_success']} "
              f"f9={m.get('finger9_contact_steps')} f10={m.get('finger10_contact_steps')} "
              f"overlap={m.get('two_finger_overlap_steps')} "
              f"slip={m.get('slip_after_release_m'):.5f} "
              f"ang_tail={m.get('ang_speed_tail_max_rad_s'):.3f} reason={att.record['failure_reason']}")
    summary = {
        'eval_version': rows[0]['eval_version'] if rows else None,
        'n': len(rows),
        'n_success': sum(1 for r in rows if r['strict_success']),
        'n_both_fingers': sum(1 for r in rows if r['checks'].get('grasped_by_two_fingers')),
        'n_two_fingers_simultaneous': sum(1 for r in rows
                                          if r['checks'].get('two_fingers_simultaneous')),
        'two_finger_overlap_steps': [r.get('two_finger_overlap_steps') for r in rows],
        'n_slip_ok': sum(1 for r in rows if r['checks'].get('no_slip_after_release')),
        'n_ang_tail_ok': sum(1 for r in rows if r['checks'].get('ang_speed_ok')),
        'rows': rows,
        'note': ('旧记录无法决定"双指/滑移"（links 汇总了 link_a，且未保存尾段轨迹）；'
                 '本文件用 v1.1 评价器重放抽样直接测量，作为对已有数据质量的外部证据，'
                 '不修改旧记录中的 strict_success。'),
    }
    (out / 'reeval_sample.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                                       default=float), encoding='utf-8')
    print(json.dumps({k: summary[k] for k in ('n', 'n_success', 'n_both_fingers', 'n_slip_ok',
                                              'n_ang_tail_ok')}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
