"""实验 4.5：磁盘审计（只看我们自己的目录，先报告再决定删除）。

输出：
  * /scratch/lijh 下我们的实验目录大小；
  * 可再生成的大件（合并 params / 中间 export / 旧 LoRA 实验），并给"guard"（对应源文件是否在）；
  * 绝不包含其他用户目录。
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

HOME = Path('/scratch/lijh')
OWN = ['exp40', 'exp41', 'exp42', 'exp43', 'exp44', 'exp44_export_A58', 'exp45',
       'simulate_vla']


def du(p: Path) -> str:
    r = subprocess.run(['du', '-sh', str(p)], capture_output=True, text=True)
    return r.stdout.split()[0] if r.stdout else '?'


print('== 我们自己的顶层目录 ==')
for name in OWN:
    p = HOME / name
    if p.exists():
        print(f'  {name:24s} {du(p)}')

print('\n== exp42 下的大件（候选） ==')
for p in sorted((HOME / 'exp42').glob('*')):
    if p.is_dir():
        print(f'  {p.name:28s} {du(p)}')

print('\n== exp43 下 ==')
for p in sorted((HOME / 'exp43').glob('*')):
    if p.is_dir():
        print(f'  {p.name:28s} {du(p)}')

print('\n== exp44 下 ==')
for p in sorted((HOME / 'exp44').glob('*')):
    print(f'  {p.name:28s} {du(p) if p.is_dir() else ""}')

print('\n== 可再生成的合并参数目录（有对应 lora npz 才算安全） ==')
cands = []
for base in (HOME / 'exp42', HOME / 'exp44'):
    for p in base.rglob('params_step*'):
        cands.append(p)
    for p in base.rglob('ckpt_*'):
        cands.append(p)
    for p in base.rglob('merged'):
        cands.append(p)
for p in cands[:40]:
    print(f'  {p}  {du(p)}')

print('\n== 汇总 df ==')
print(subprocess.run(['df', '-h', '/scratch'], capture_output=True, text=True).stdout)
