"""分析 Gate1 trace：adapter 目标 vs 专家目标、clip 触发、锚点漂移。"""
import collections
import csv
import json
import os
import sys

import numpy as np

path = sys.argv[1]
rows = [r for r in csv.DictReader(open(path, encoding='utf-8')) if r['mode'] == 'adapter']
exp = [r for r in csv.DictReader(open(path, encoding='utf-8')) if r['mode'] == 'expert']


def vec(s):
    return np.array([float(x) for x in s.split(';')])


E = np.stack([vec(r['expert_target_q']) for r in rows])
T = np.stack([vec(r['adapter_target_q']) for r in rows])
Q = np.stack([vec(r['actual_q']) for r in rows])
A = np.stack([vec(r['anchor_q']) for r in rows])
d = np.abs(T - E)
QE = np.stack([vec(r['actual_q']) for r in exp]) if exp else None

print('n_ticks', len(rows), 'chunks', len({r['chunk_id'] for r in rows}))
print('max |adapter_target - expert_target|', round(float(d.max()), 4))
print('max |actual_q - expert_state|(adapter)', round(float(np.abs(Q - np.stack([vec(r['actual_q']) for r in exp])).max()), 4) if QE is not None else None)
# 每 chunk 内 k=0..7 的 target 误差
by = collections.defaultdict(dict)
for i, r in enumerate(rows):
    by[int(r['chunk_id'])][int(r['k_in_chunk'])] = i
ks = sorted({k for c in by.values() for k in c})
err_by_k = {}
for k in ks:
    idx = [c[k] for c in by.values() if k in c]
    err_by_k[k] = round(float(d[idx].max()), 4)
print('target_err_by_k', json.dumps(err_by_k))
# 锚点漂移：anchor 与专家记录 q 的差（用 expert 行做参考）
if QE is not None:
    print('anchor_vs_expert_state_max', round(float(np.abs(A - QE).max()), 4))
# 触发 0.25 clip 的比例：|target - anchor| > 0.25
delta = np.abs(T - A)
print('frac |target-anchor| > 0.25 =', round(float((delta > 0.25).mean()), 4),
      ' max', round(float(delta.max()), 4))
print('expert delta from anchor max', round(float(np.abs(E - A).max()), 4))
