import json
import re
from pathlib import Path

log = Path('/scratch/lijh/exp45/resume_test.log')
txt = log.read_text(errors='ignore')
# 分成 run1/run2 两段
parts = re.split(r'=== run\d', txt)
runs = [p for p in parts if p.strip()]
print('segments', len(runs))
seqs = []
for seg in runs:
    losses = re.findall(r'step (\d+)/(\d+) loss=([\d.]+)', seg)
    seqs.append({int(s): float(l) for s, _, l in losses})
for i, s in enumerate(seqs):
    print(f'run{i+1} steps', sorted(s)[:3], '...', sorted(s)[-3:])
if len(seqs) >= 2 and seqs[0] and seqs[1]:
    common = sorted(set(seqs[0]) & set(seqs[1]))
    print('common steps', common)
    diffs = [(k, seqs[0][k], seqs[1][k]) for k in common]
    for k, a, b in diffs:
        print(f'  step {k}: continuous={a:.6f} resumed={b:.6f} |diff|={abs(a-b):.2e}')
# 也读各自的 training_curve.csv
for i in (1, 2):
    p = Path(f'/scratch/lijh/exp45/resume_test/run{i}/training_curve.csv')
    if p.exists():
        lines = p.read_text().strip().splitlines()[1:]
        print(f'run{i} curve rows', len(lines), lines[:2], lines[-2:])
