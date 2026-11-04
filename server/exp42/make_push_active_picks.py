"""生成"B90 主动推动"帧的 picks（机器人接触 + 未触墙 + 专家 |dq|>0.01）。"""
import csv
import json
import sys

import numpy as np

ROOT = '/scratch/lijh/exp40/repo/results/experiment_4_2/contract_replay_p0c'
path = f'{ROOT}/expert_vs_adapter_trace_B90_measured.csv'
rows = [r for r in csv.DictReader(open(path, encoding='utf-8')) if r['mode'] == 'adapter']


def vec(s):
    return np.array([float(x) for x in s.split(';')])


items = []
for ep in (0, 1):
    sub = [r for r in rows if int(r['episode']) == ep]
    for r in sub:
        dq = float(np.abs(vec(r['expert_target_q']) - vec(r['actual_q'])).max())
        if int(r['robot_block_contacts']) > 0 and int(r['block_wall_contacts']) == 0 and dq > 0.01:
            items.append({'skill': 'B90', 'episode': ep, 'tick': int(r['tick']),
                          'phase': 'push_active', 'robot_block': int(r['robot_block_contacts']),
                          'block_wall': 0, 'frame': int(r['tick'])})
out = {'items': items, 'note': '机器人接触+未触墙+专家|dq|>0.01 的主动推动帧'}
dest = f'{ROOT}/expert_state_picks_push_active.json'
open(dest, 'w', encoding='utf-8').write(json.dumps(out, ensure_ascii=False, indent=2))
print('wrote', dest, 'items', len(items),
      'episodes', sorted({i['episode'] for i in items}))
