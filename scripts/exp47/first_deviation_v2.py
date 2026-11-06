"""实验 4.7 Gate 4：用**同一运行时路径**的 EE 重算首次偏离（取代 4.6 的 FK 参考）。

专家 EE：gate3 实验 A（逐周期专家回放）trace 的 ee_before（每个 20 Hz tick 一行）。
rollout EE：实验 4.6 step3000 的 substeps（每 tick 取 k=0 子步）。
输出 first_deviation_v2.csv：tick 0 距离（应为 mm 级）、首次 >3cm 且持续的 tick、最大偏离。
"""
from __future__ import annotations

import csv
import json

import numpy as np
import re
from pathlib import Path

REPO = Path('/scratch/lijh/exp40/repo')
OUT = REPO / 'results' / 'experiment_4_7'
EXPERT_TRACE = OUT / 'expert_execution_trace.csv'
ROLL = REPO / 'results' / 'experiment_4_6' / 'rollouts'


def main() -> int:
    expert = {}
    for r in csv.DictReader(open(EXPERT_TRACE, encoding='utf-8')):
        if r['mode'] != 'A' or r['safe'] != 'True':
            continue
        key = (r['skill'], int(r['episode']))
        expert.setdefault(key, {})[int(r['tick'])] = np_vec(r['ee_before'])

    rows = []
    for f in sorted(ROLL.glob('step3000/off*/*_substeps.csv')):
        m = re.search(r'off(\d+)/(A90|B90)_ep(\d+)_substeps', str(f))
        if not m:
            continue
        off, skill, ep = int(m.group(1)), m.group(2), int(m.group(3))
        exp = expert.get((skill, ep))
        if not exp:
            continue
        first = {}
        for r in csv.DictReader(open(f, encoding='utf-8')):
            t = int(r['tick'])
            if t not in first:
                first[t] = np_vec(r['ee_before'])
        devs = []
        for t in sorted(first):
            if t in exp:
                devs.append((t, float(np.linalg.norm(first[t] - exp[t]))))
        if not devs:
            continue
        first_tick_dev = devs[0][1]
        first_over = None
        for i, (t, d) in enumerate(devs):
            if d > 0.03:
                rest = [x[1] for x in devs[i:]]
                if sum(1 for x in rest if x > 0.01) / len(rest) >= 0.8:
                    first_over = t
                    break
        rows.append({'skill': skill, 'episode': ep, 'seed_offset': off,
                     'tick0_ee_dist_m': round(first_tick_dev, 5),
                     'max_ee_dist_m': round(max(d for _, d in devs), 4),
                     'first_gt3cm_sustained_tick': first_over,
                     'n_ticks': len(devs)})
    with open(OUT / 'first_deviation_v2.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (OUT / 'first_deviation_v2.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                                                 encoding='utf-8')
    print(json.dumps(rows, ensure_ascii=False, indent=1))
    return 0


def np_vec(s: str):
    import numpy as np
    return np.asarray([float(x) for x in s.split(';')], dtype=float)


if __name__ == '__main__':
    raise SystemExit(main())
