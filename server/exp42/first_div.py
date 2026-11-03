"""定位 Gate1 adapter 回放的第一次明显偏离。"""
import csv
import sys

import numpy as np

path = sys.argv[1]
rows = [r for r in csv.DictReader(open(path, encoding='utf-8')) if r['mode'] == 'adapter']


def vec(s):
    return np.array([float(x) for x in s.split(';')])


prev = None
first = None
for r in rows:
    E, T, A = vec(r['expert_target_q']), vec(r['adapter_target_q']), vec(r['anchor_q'])
    err = float(np.abs(T - E).max())
    raw = float(np.abs(E - A).max())
    if err > 0.1 and first is None:
        first = (r['tick'], r['chunk_id'], r['k_in_chunk'], err, raw,
                 np.round(E - A, 3).tolist())
    if first is not None and int(r['tick']) < int(first[0]) + 6:
        print(f"t={r['tick']:>4} chunk={r['chunk_id']:>3} k={r['k_in_chunk']} "
              f"|T-E|={err:.4f} |E-A|={raw:.4f}")
print('FIRST >0.1:', first)
print('total rows', len(rows))
