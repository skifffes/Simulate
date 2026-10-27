"""实验 2.4 Q8：D0 / D1 / D2 在相同 G0 seeds 上的接触行为对照（各 5 集诊断）。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results/experiment_2_4'
SEEDS = [40000000, 40000001, 40000002, 40000003, 40000004]


def load(path: Path):
    out = {}
    for seed in SEEDS:
        p = path / 'G0_a0b_direct' / f'seed_{seed}' / 'summary.json'
        if p.exists():
            out[seed] = json.loads(p.read_text())
    return out


def stats(d):
    if not d:
        return None
    vals = list(d.values())
    return {
        'n': len(vals),
        'success': sum(1 for v in vals if v['success']),
        'grasp_lift': sum(1 for v in vals if v['both_fingers'] and v['table_loss']
                          and v['max_lift_m'] > 0.03),
        'both_fingers': sum(1 for v in vals if v['both_fingers']),
        'table_loss': sum(1 for v in vals if v['table_loss']),
        'wall_contact': sum(1 for v in vals if v['wall_contact']),
        'max_lift_mean': round(float(np.mean([v['max_lift_m'] for v in vals])), 4),
        'seeds': sorted(d),
    }


def main():
    out = {
        'D2_formal': stats(load(RES)),
        'D0_diag': stats(load(RES / 'models_D0')),
        'D1_diag': stats(load(RES / 'models_D1')),
        'note': 'all restricted to seeds 40000000-40000004; D2 from the 10-episode G0 formal run, '
                'D0/D1 from 5-episode diagnostic runs',
    }
    (RES / 'model_compare.json').write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                            encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
