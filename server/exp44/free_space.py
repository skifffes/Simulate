"""磁盘腾挪：删除**可再生成**的合并参数目录与已评测的 opt state，保留 LoRA 权重与数据集。

规则（先核实再删）：
  1. `/scratch/lijh/exp42/curve2400/ckpts/params_step02400` 只有在对应
     `lora_step02400.npz` 存在时才删除（合并可按需再导出）。
  2. `/scratch/lijh/exp44/{A,B}/ckpts/opt_step06000.npz` 仅在 12000 的 opt state
     已存在、或明确记录用途后删除；此处先保留，只打印占用。
  3. 任何数据集、LoRA、logs 都不删。
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

cands = [
    (Path('/scratch/lijh/exp42/curve2400/ckpts/params_step02400'),
     Path('/scratch/lijh/exp42/curve2400/ckpts/lora_step02400.npz')),
    (Path('/scratch/lijh/exp42/curve1800/ckpt_1800'),
     Path('/scratch/lijh/exp42/curve1800/ckpts/lora_step01800.npz')),
]
for target, guard in cands:
    if not target.exists():
        print('skip (missing):', target)
        continue
    if not guard.exists():
        print('REFUSE (no guard lora):', target)
        continue
    size = subprocess.run(['du', '-sh', str(target)], capture_output=True, text=True).stdout.strip()
    print('would delete', target, size, '| guard ok:', guard.name)
    shutil.rmtree(target)
    print('deleted', target)

for p in (Path('/scratch/lijh/exp44/A/ckpts'), Path('/scratch/lijh/exp44/B/ckpts')):
    tot = sum(f.stat().st_size for f in p.glob('*.npz'))
    print(p, round(tot / 1e9, 1), 'GB in', len(list(p.glob('*.npz'))), 'npz')
print(subprocess.run(['df', '-h', '/scratch'], capture_output=True, text=True).stdout)
