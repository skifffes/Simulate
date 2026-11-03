"""统计关节角包裹（wrap）情况：|A[t]-S[t]| 中出现 >1 rad 的维度/次数。"""
import json
import os

import numpy as np

DATA = os.environ.get('EXP42_DATA', '/scratch/lijh/exp42/data')
meta = [json.loads(l) for l in open(os.path.join(DATA, 'meta.jsonl'), encoding='utf-8')
        if l.strip()]
tot = 0
bad_frames = 0
per_dim = np.zeros(7, dtype=int)
max_abs = 0.0
eps_with = 0
for m in meta:
    z = np.load(os.path.join(DATA, m['state_path']))
    S = z['state'].astype(float)
    A = z['action'].astype(float)
    d = A[:, :7] - S[:, :7]
    n_bad = int((np.abs(d) > 1.0).any(axis=1).sum())
    if n_bad:
        eps_with += 1
    bad_frames += n_bad
    tot += d.shape[0]
    per_dim += (np.abs(d) > 1.0).sum(axis=0)
    max_abs = max(max_abs, float(np.abs(d).max()))
print(json.dumps({
    'n_episodes': len(meta), 'n_frames': tot,
    'episodes_with_abs_delta_gt_1rad': eps_with,
    'frames_with_abs_delta_gt_1rad': bad_frames,
    'frac_frames': round(bad_frames / max(tot, 1), 5),
    'per_joint_count': per_dim.tolist(),
    'max_abs_delta_rad': max_abs,
}, ensure_ascii=False, indent=2))
