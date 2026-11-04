"""检查 B90 专家 trace 里是否存在"机器人接触 + 未触墙 + 专家确实在动"的主动推动帧。"""
import csv
import sys

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else (
    '/scratch/lijh/exp40/repo/results/experiment_4_2/contract_replay_p0c/'
    'expert_vs_adapter_trace_B90_measured.csv')
rows = [r for r in csv.DictReader(open(path, encoding='utf-8')) if r['mode'] == 'adapter']


def vec(s):
    return np.array([float(x) for x in s.split(';')])


for ep in (0, 1):
    sub = [r for r in rows if int(r['episode']) == ep]
    dq = []
    for r in sub:
        tgt = vec(r['expert_target_q'])
        act = vec(r['actual_q'])
        dq.append(float(np.abs(tgt - act).max()))
    dq = np.array(dq)
    rb = np.array([int(r['robot_block_contacts']) > 0 for r in sub])
    wl = np.array([int(r['block_wall_contacts']) > 0 for r in sub])
    sel = np.where(rb & ~wl)[0]
    print(f'ep{ep}: ticks={len(sub)} robot_contact_no_wall={len(sel)} '
          f'|dq| in that set: max={dq[sel].max() if len(sel) else 0:.4f} '
          f'median={np.median(dq[sel]) if len(sel) else 0:.4f} '
          f'frac>0.01={float((dq[sel] > 0.01).mean()) if len(sel) else 0:.2f}')
    active = np.where(rb & ~wl & (dq > 0.01))[0]
    print('   active push ticks (sample):', active[:10].tolist(),
          'expert |dq|:', np.round(dq[active][:10], 4).tolist())
