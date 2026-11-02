"""快速查看实验 4.1 采集进度（不需要 poweshell 转义）。"""
import json
import os
import subprocess
import sys

repo = '/scratch/lijh/exp40/repo'
for s in ('A90', 'B90'):
    p = f'{repo}/data/exp41/{s}/meta/episodes.jsonl'
    if os.path.exists(p):
        n = sum(1 for _ in open(p, encoding='utf-8'))
        print(s, 'episodes', n)
    else:
        print(s, 'not yet')
print(subprocess.run(['bash', '-lc', 'ps -u lijh -o etime,args | grep collect41 | grep -v grep | head -2'],
                     capture_output=True, text=True).stdout)
