"""RQ6：严格评价器 vs 旧版 v2.2 判据的标签差异（混淆表）。

旧 v2.2（`scripts/eval_multi_atomic_vla.py:120-128`）：
    ok = |gap| ≤ 0.02 且 |dy| ≤ 0.02 且 dyaw ≤ 15° 且 tilt ≤ 12° 且 speed < 0.01
**没有任何接触要求**，所以"停在墙前 2 cm 内"就算成功。

本脚本对每条已记录的 B90 尝试，用同一批指标同时计算两种标签，并给出混淆表与差异原因。

用法：
    python scripts/exp41/compare_legacy_eval.py --records results/experiment_4_1/dataset/records \
        --out results/experiment_4_1/evaluation/legacy_vs_strict.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def legacy_v22(m: dict) -> tuple[bool, dict]:
    gap = m.get('terminal_gap_m')
    dy = m.get('lateral_drift_m')
    dyaw = m.get('yaw_change_rad')
    tilt = m.get('tilt_deg')
    speed = m.get('final_speed_m_s')
    if None in (gap, dy, dyaw, tilt, speed):
        return False, {'error': 'missing_metrics'}
    ok = (abs(gap) <= 0.02 and abs(dy) <= 0.02 and dyaw <= math.radians(15)
          and tilt <= 12 and speed < 0.01)
    return bool(ok), {'gap': gap, 'dy': dy, 'dyaw_deg': math.degrees(dyaw), 'tilt': tilt,
                      'speed': speed}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--records', default='results/experiment_4_1/dataset/records')
    ap.add_argument('--out', default='results/experiment_4_1/evaluation/legacy_vs_strict.json')
    args = ap.parse_args(argv)
    recs = []
    for f in sorted(Path(args.records).glob('*.json')):
        r = json.loads(f.read_text(encoding='utf-8'))
        if r.get('skill') == 'B90':
            recs.append(r)
    rows = []
    for r in recs:
        m = r['metrics']
        old_ok, detail = legacy_v22(m)
        rows.append({
            'attempt_id': r['attempt_id'], 'config_id': r.get('config_id'),
            'strict_success': bool(r['strict_success']),
            'strict_failure_reason': r['failure_reason'],
            'legacy_v22_success': old_ok,
            'legacy_detail': detail,
            'wall_contact_steps': m.get('wall_contact_steps'),
            'wall_contact_force_max_n': m.get('wall_contact_force_max_n'),
        })
    n = len(rows)
    both = sum(1 for r in rows if r['strict_success'] and r['legacy_v22_success'])
    strict_only = sum(1 for r in rows if r['strict_success'] and not r['legacy_v22_success'])
    legacy_only = sum(1 for r in rows if not r['strict_success'] and r['legacy_v22_success'])
    neither = sum(1 for r in rows if not r['strict_success'] and not r['legacy_v22_success'])
    table = {'n_b90_attempts': n, 'both_success': both, 'strict_only': strict_only,
             'legacy_only': legacy_only, 'neither': neither,
             'legacy_success_rate': round((both + legacy_only) / max(1, n), 3),
             'strict_success_rate': round((both + strict_only) / max(1, n), 3),
             'legacy_only_examples': [r for r in rows if r['legacy_v22_success']
                                      and not r['strict_success']][:10],
             'rows': rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(table, ensure_ascii=False, indent=2, default=float),
                              encoding='utf-8')
    print(json.dumps({k: table[k] for k in ('n_b90_attempts', 'both_success', 'strict_only',
                                            'legacy_only', 'neither', 'legacy_success_rate',
                                            'strict_success_rate')}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
