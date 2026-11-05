"""训练清单 v3：在 export 的 314 条 train 上，剔除旧 B90 的 3 条 v1.2 失败项（311 条）。

同时做"动作-状态时序一致性"抽查：比较 |a[t]-s[t]| 与 |a[t]-s[t+1]|（若后者更小，
说明动作更像"下一状态目标"；两者接近说明动作是"目标命令"而非下一帧状态）。
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

EXPORT = Path('/scratch/lijh/exp43/export')
L = Path('/scratch/lijh/exp40/repo/results/experiment_4_3/data_expansion')
OLD_FAIL = {('B90', 8), ('B90', 20), ('B90', 29)}


def main() -> int:
    meta = [json.loads(l) for l in (EXPORT / 'meta.jsonl').read_text(
        encoding='utf-8').splitlines() if l.strip()]
    train = [m for m in meta if m.get('split') == 'train']
    kept, excluded = [], []
    for m in train:
        ep = int(m['episode_index'])
        if ep < 1000 and (m['skill'], ep) in OLD_FAIL:
            excluded.append({'skill': m['skill'], 'episode_index': ep,
                             'reason': 'v1.2 recheck failed (old data)'})
            continue
        kept.append({'skill': m['skill'], 'episode_index': ep,
                     'source': m.get('source', 'exp41'),
                     'seed': m.get('seed'), 'config_id': m.get('config_id'),
                     'n_frames': m.get('n_frames'), 'prompt': m.get('prompt'),
                     'state_path': m['state_path'], 'top_path': m['top_path'],
                     'wrist_path': m['wrist_path']})
    out = {
        'version': 'exp43-train-manifest-v3',
        'basis': 'export/meta.jsonl train 314 minus old-B90 v1.2 failures',
        'n_episodes': len(kept),
        'n_by_skill': dict(Counter(e['skill'] for e in kept)),
        'n_by_source': dict(Counter(e['source'] for e in kept)),
        'excluded': excluded,
        'episodes': kept,
    }
    (L / 'train_manifest_v3.json').write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                               encoding='utf-8')

    # 动作-时序一致性抽查（20 条）
    rows = []
    for e in kept[::max(1, len(kept) // 20)]:
        z = np.load(EXPORT / e['state_path'])
        s, a = z['state'].astype(float), z['action'].astype(float)
        n = min(len(s), len(a))
        if n < 3:
            continue
        d_same = np.abs(a[:n - 1, :7] - s[:n - 1, :7]).mean()
        d_next = np.abs(a[:n - 1, :7] - s[1:n, :7]).mean()
        rows.append({'skill': e['skill'], 'ep': e['episode_index'],
                     'a_vs_s_current': float(d_same), 'a_vs_s_next': float(d_next),
                     'next_smaller': bool(d_next < d_same)})
    (L / 'action_timing_check.json').write_text(
        json.dumps({'n': len(rows), 'rows': rows,
                    'frac_next_smaller': float(np.mean([r['next_smaller'] for r in rows]))},
                   ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: out[k] for k in ('n_episodes', 'n_by_skill', 'n_by_source', 'excluded')},
                     ensure_ascii=False))
    print('timing: frac where |a-s_next| < |a-s_now| =',
          round(float(np.mean([r['next_smaller'] for r in rows])), 2),
          'median a-s_now', round(float(np.median([r['a_vs_s_current'] for r in rows])), 4),
          'median a-s_next', round(float(np.median([r['a_vs_s_next'] for r in rows])), 4))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
