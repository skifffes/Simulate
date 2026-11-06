"""实验 4.5：汇总配对闭环结果 → paired_rollout_results.csv + 分技能成功率。"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

L = Path('/scratch/lijh/exp40/repo/results/experiment_4_5')
ROLL = L / 'rollouts'


def main() -> int:
    rows = []
    for d in sorted(ROLL.glob('*_step*')):
        m = re.match(r'(ae|lora)_step(\d+)', d.name)
        if not m:
            continue
        model, step = m.group(1), int(m.group(2))
        for off_dir in sorted(d.glob('off*')):
            m2 = re.match(r'off(\d+)', off_dir.name)
            if not m2:
                continue
            off = int(m2.group(1))
            for f in sorted(off_dir.glob('*.json')):
                if f.name.endswith('_summary.json'):
                    continue
                j = json.loads(f.read_text(encoding='utf-8'))
                mm = j.get('metrics') or {}
                rows.append({
                    'model': model, 'step': step, 'seed_offset': off,
                    'skill': j.get('skill'), 'episode': j.get('episode'),
                    'success': bool(j.get('strict_success')),
                    'failure_reason': j.get('failure_reason') or '',
                    'advance_x_m': mm.get('advance_x_m'),
                    'terminal_gap_m': mm.get('terminal_gap_m'),
                    'wall_contact_steps': mm.get('wall_contact_steps'),
                    'robot_block_contact_steps': mm.get('robot_block_contact_steps'),
                    'tilt_deg': mm.get('tilt_deg'),
                    'lift_m': mm.get('lift_m'),
                    'position_error_m': mm.get('position_error_m'),
                    'two_finger_overlap_steps': mm.get('two_finger_overlap_steps'),
                    'wall_force_max_n': mm.get('wall_contact_force_max_n'),
                })
    keys = ['model', 'step', 'seed_offset', 'skill', 'episode', 'success', 'failure_reason',
            'advance_x_m', 'terminal_gap_m', 'wall_contact_steps',
            'robot_block_contact_steps', 'tilt_deg', 'lift_m', 'position_error_m',
            'two_finger_overlap_steps', 'wall_force_max_n']
    with open(L / 'paired_rollout_results.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    summary = {'n_rollouts': len(rows), 'by_model': {}}
    for model in sorted({r['model'] for r in rows}):
        sel = [r for r in rows if r['model'] == model]
        per_skill = {}
        for skill in ('A90', 'B90'):
            ss = [r for r in sel if r['skill'] == skill]
            per_skill[skill] = {
                'n': len(ss), 'success': sum(1 for r in ss if r['success']),
                'failure_reasons': dict(Counter(r['failure_reason'] for r in ss if not r['success'])),
            }
        summary['by_model'][model] = {
            'n': len(sel), 'success': sum(1 for r in sel if r['success']),
            'per_skill': per_skill,
            'median_advance_b90': sorted([r['advance_x_m'] for r in sel
                                          if r['skill'] == 'B90' and r['advance_x_m'] is not None])[
                len([r for r in sel if r['skill'] == 'B90' and r['advance_x_m'] is not None]) // 2]
            if any(r['skill'] == 'B90' and r['advance_x_m'] is not None for r in sel) else None,
        }
    (L / 'rollout_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                             encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
