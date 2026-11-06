import csv
from collections import defaultdict

import numpy as np

rows = list(csv.DictReader(open('/scratch/lijh/exp40/repo/results/experiment_4_7/'
                                 'expert_execution_trace.csv', encoding='utf-8')))
print('rows', len(rows))
by = defaultdict(list)
for r in rows:
    by[(r['mode'], r['safe'])].append(r)
for k, v in sorted(by.items()):
    tr = np.array([float(x['track_err_max']) for x in v])
    ad = np.array([float(x['adapter_target_vs_expert']) for x in v])
    rl = sum(1 for x in v if x['rate_limited'] == 'True')
    acc = sum(1 for x in v if x['accepted'] == 'True')
    grip = np.array([abs(float(x['gripper_cmd']) - float(x['gripper_actual']))
                     for x in v])
    print(f'{k}: n={len(v)} track_max={tr.max():.5f} track_mean={tr.mean():.5f} '
          f'adapter_max={ad.max():.5f} rate_limited={rl} accepted={acc} '
          f'grip_track_max={grip.max():.4f}')
# 每个 episode 的模式 A track
for r in rows[:1]:
    pass
