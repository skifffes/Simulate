"""逐 seed 的余弦分布：判断 2400 的"退化"是系统性还是种子组不巧。

对 1800 与 2400 各读两个 npz（seeds 0-3 与 4-11），按 (skill, phase) 统计
**每个 seed 的 cos_joint 均值**，并给出 12 个 seed 的分布（mean/median/min/max/std）。
"""
import itertools
import json
from pathlib import Path

import numpy as np

L = Path(r'results\experiment_4_2\contract_replay_p0c')
PAIRS = {
    1800: [L / 'expert_state_curve1800.npz', L / 'expert_state_curve1800_s4.json.npz'],
    2400: [L / 'expert_state_curve2400.npz', L / 'expert_state_curve2400_s4.json.npz'],
}
MIN_EXPERT = 0.01


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def load(paths):
    rows = []
    for p in paths:
        z = np.load(p, allow_pickle=False)
        for i in range(len(z['tick'])):
            ef, mf = z['expert_first'][i], z['model_first'][i]
            rows.append({'skill': str(z['skill'][i]), 'phase': str(z['phase'][i]),
                         'seed': int(z['seed'][i]),
                         'cos': (float(unit(mf) @ unit(ef))
                                 if np.linalg.norm(ef) > MIN_EXPERT else None),
                         'absmax': float(np.abs(mf).max()),
                         'mae': float(np.abs(mf - ef).mean())})
    return rows


def main():
    out = {}
    for step, paths in PAIRS.items():
        rows = load(paths)
        keys = sorted({(r['skill'], r['phase']) for r in rows})
        per = {}
        for skill, phase in keys:
            sel = [r for r in rows if r['skill'] == skill and r['phase'] == phase]
            by_seed = {}
            for seed, group in itertools.groupby(sorted(sel, key=lambda r: r['seed']),
                                                  key=lambda r: r['seed']):
                g = list(group)
                cs = [x['cos'] for x in g if x['cos'] is not None]
                by_seed[int(seed)] = {
                    'cos': float(np.mean(cs)) if cs else None,
                    'absmax': float(np.mean([x['absmax'] for x in g])),
                    'mae': float(np.mean([x['mae'] for x in g])),
                }
            cs_all = [v['cos'] for v in by_seed.values() if v['cos'] is not None]
            per[f'{skill}:{phase}'] = {
                'n_states': len(sel) // max(len(by_seed), 1),
                'per_seed_cos': by_seed,
                'cos_mean': float(np.mean(cs_all)) if cs_all else None,
                'cos_median': float(np.median(cs_all)) if cs_all else None,
                'cos_min': float(np.min(cs_all)) if cs_all else None,
                'cos_max': float(np.max(cs_all)) if cs_all else None,
                'cos_std': float(np.std(cs_all)) if cs_all else None,
                'absmax_mean': float(np.mean([v['absmax'] for v in by_seed.values()])),
                'mae_mean': float(np.mean([v['mae'] for v in by_seed.values()])),
            }
        out[str(step)] = per

    (L / 'seed_variance_analysis.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f"{'phase':22s} {'1800 cos(mean/min/max/std)':>34s}   {'2400 cos(mean/min/max/std)':>34s}")
    for k in out['1800']:
        a, b = out['1800'][k], out['2400'][k]
        if a['cos_mean'] is None:
            continue
        print(f"{k:22s} {a['cos_mean']:6.2f}/{a['cos_min']:5.2f}/{a['cos_max']:5.2f}/{a['cos_std']:.2f}"
              f"   {'':>10s} {b['cos_mean']:6.2f}/{b['cos_min']:5.2f}/{b['cos_max']:5.2f}/{b['cos_std']:.2f}")
    print('\nper-seed cos (B90:approach):')
    for step in ('1800', '2400'):
        print(step, {s: round(v['cos'], 2) if v['cos'] is not None else None
                     for s, v in out[step]['B90:approach']['per_seed_cos'].items()})


if __name__ == '__main__':
    main()
