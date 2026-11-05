import glob
import json
import os
import re
import subprocess

# 找训练日志（含 step 行）
cands = []
for pat in ('/scratch/lijh/exp42/**/*.log', '/scratch/lijh/exp42/*.txt',
            '/scratch/lijh/exp42/**/*.json'):
    cands.extend(glob.glob(pat, recursive=True))
print('n candidates', len(cands))
for f in sorted(set(cands))[:40]:
    print(f, round(os.path.getsize(f) / 1e6, 2), 'MB')

# 训练目录里的 manifest/log
for d in ('/scratch/lijh/exp42/curve2400', '/scratch/lijh/exp42/curve1800',
          '/scratch/lijh/exp42/run2'):
    print('---', d)
    for f in sorted(glob.glob(d + '/*')):
        print('  ', os.path.basename(f), round(os.path.getsize(f) / 1e6, 2), 'MB')
        if f.endswith('.json') and 'ckpt' not in f:
            try:
                j = json.load(open(f))
                print('    keys', list(j)[:12])
                if 'elapsed_s' in j or 'throughput' in j:
                    print('    ', {k: j[k] for k in j if 'elapsed' in k or 'per_step' in k or 'through' in k})
            except Exception:
                pass
        if f.endswith('.log') or f.endswith('.txt'):
            txt = open(f, errors='ignore').read()
            steps = re.findall(r'step\s+(\d+)\s+.*?loss=([\d.]+)', txt)
            times = re.findall(r'\[(\d{2}:\d{2}:\d{2})\]', txt)
            print('    step lines', len(steps), 'first', steps[:2], 'time markers', len(times))
