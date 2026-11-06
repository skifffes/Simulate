"""实验 4.5 补充：训练初态配对闭环统计（rollouts_traininit，支持 tag 后缀）。"""
from __future__ import annotations

import collections
import csv
import json
import re
import statistics
from pathlib import Path

BASE = Path('/scratch/lijh/exp40/repo/results/experiment_4_5/rollouts_traininit')
OUT = Path('/scratch/lijh/exp40/repo/results/experiment_4_5/paired_traininit_results.csv')


def main() -> int:
    rows = []
    for d in sorted(BASE.glob('*_step*')):
        m = re.match(r'([a-z]+)_step(\d+)(.*)$', d.name)
        if not m:
            continue
        model, step, tag = m.group(1), int(m.group(2)), m.group(3) or ''
        for off_dir in sorted(d.glob('off*')):
            off = int(off_dir.name.replace('off', ''))
            for f in sorted(off_dir.glob('*.json')):
                if f.name.endswith('_summary.json'):
                    continue
                j = json.loads(f.read_text(encoding='utf-8'))
                mm = j.get('metrics') or {}
                rows.append({'model': model, 'step': step, 'tag': tag, 'seed_offset': off,
                             'skill': j.get('skill'), 'episode': j.get('episode'),
                             'success': bool(j.get('strict_success')),
                             'failure_reason': j.get('failure_reason') or '',
                             'advance_x_m': mm.get('advance_x_m'),
                             'terminal_gap_m': mm.get('terminal_gap_m'),
                             'wall_contact_steps': mm.get('wall_contact_steps'),
                             'robot_block_contact_steps': mm.get('robot_block_contact_steps'),
                             'tilt_deg': mm.get('tilt_deg'), 'lift_m': mm.get('lift_m'),
                             'position_error_m': mm.get('position_error_m'),
                             'two_finger_overlap_steps': mm.get('two_finger_overlap_steps')})
    keys = ['model', 'step', 'tag', 'seed_offset', 'skill', 'episode', 'success',
            'failure_reason', 'advance_x_m', 'terminal_gap_m', 'wall_contact_steps',
            'robot_block_contact_steps', 'tilt_deg', 'lift_m', 'position_error_m',
            'two_finger_overlap_steps']
    with open(OUT, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print('rows', len(rows), '->', OUT)
    for model, tag in sorted({(r['model'], r['tag']) for r in rows}):
        for skill in ('A90', 'B90'):
            sel = [r for r in rows if r['model'] == model and r['tag'] == tag
                   and r['skill'] == skill]
            n_ok = sum(1 for r in sel if r['success'])

            def stat(k):
                v = [float(r[k]) for r in sel if r[k] is not None]
                return ((round(statistics.mean(v), 4), round(statistics.median(v), 4))
                        if v else None)

            print(f'{model}{tag} {skill}: n={len(sel)} success={n_ok} '
                  f'reasons={dict(collections.Counter(r["failure_reason"] for r in sel if not r["success"]))}')
            print(f'   advance={stat("advance_x_m")} gap={stat("terminal_gap_m")} '
                  f'wall={stat("wall_contact_steps")} rb={stat("robot_block_contact_steps")} '
                  f'lift={stat("lift_m")} tilt={stat("tilt_deg")} poserr={stat("position_error_m")}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
