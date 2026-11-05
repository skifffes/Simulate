import re
import subprocess
import time

for g in ('A', 'B'):
    f = f'/scratch/lijh/exp44/logs_{g}.txt'
    try:
        txt = open(f, errors='ignore').read().splitlines()
    except FileNotFoundError:
        print(g, 'no log')
        continue
    steps = [l for l in txt if l.startswith('step ')]
    print(f'--- {g}: {len(steps)} step-lines')
    for l in steps[-2:]:
        print('   ', l.strip())
    # 训练速度
    m = [l for l in txt if 'seconds_per_optimizer_step' in l or 'saved ckpt' in l]
    if m:
        print('   ', m[-1].strip()[:120])
print(subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                      '--format=csv,noheader'], capture_output=True, text=True).stdout)
print('now', time.strftime('%H:%M:%S'))
