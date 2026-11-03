"""数据侧统计：每条 episode 开头有多少帧"专家保持不动"，以及零动作帧占比。"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, '/scratch/lijh/exp40/repo/src')
DATA = os.environ.get('EXP42_DATA', '/scratch/lijh/exp42/data')
meta = [json.loads(l) for l in open(os.path.join(DATA, 'meta.jsonl'), encoding='utf-8')
        if l.strip()]
rows = []
for m in meta:
    z = np.load(os.path.join(DATA, m['state_path']))
    S, A = z['state'].astype(float), z['action'].astype(float)
    dq = np.abs(A[:, :7] - S[:, :7]).max(axis=1)
    lead = 0
    for v in dq:
        if v < 1e-4:
            lead += 1
        else:
            break
    rows.append({'skill': m['skill'], 'episode': m['episode_index'], 'split': m['split'],
                 'n_frames': int(len(dq)), 'leading_still_frames': lead,
                 'frac_still_lt_1e-4': float((dq < 1e-4).mean()),
                 'frac_still_lt_0.005': float((dq < 0.005).mean()),
                 'max_dq': float(dq.max())})
summ = {}
for skill in ('A90', 'B90'):
    sel = [r for r in rows if r['skill'] == skill]
    summ[skill] = {
        'n_episodes': len(sel),
        'leading_still_frames_mean': float(np.mean([r['leading_still_frames'] for r in sel])),
        'leading_still_frames_median': float(np.median([r['leading_still_frames'] for r in sel])),
        'frac_still_lt_1e-4_mean': float(np.mean([r['frac_still_lt_1e-4'] for r in sel])),
        'frac_still_lt_0.005_mean': float(np.mean([r['frac_still_lt_0.005'] for r in sel])),
        'max_dq_mean': float(np.mean([r['max_dq'] for r in sel])),
    }
print(json.dumps({'summary': summ, 'episodes': rows}, ensure_ascii=False, indent=1)[:2000])
open('/scratch/lijh/exp42/diag/still_frames.json', 'w', encoding='utf-8').write(
    json.dumps({'summary': summ, 'episodes': rows}, ensure_ascii=False, indent=2))
