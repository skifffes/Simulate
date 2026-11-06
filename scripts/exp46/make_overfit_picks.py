"""实验 4.6：为 4 条过拟合 episode 生成**逐帧**评测 picks（每 2 帧取 1，seeds 0/1）。

用于回答"模型是否在全部专家状态上都能准确预测"，覆盖夹爪闭合、接触与整段 16-step chunk。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DATA = Path('/scratch/lijh/exp46/export_overfit')
OUT = Path('/scratch/lijh/exp40/repo/results/experiment_4_6/overfit_picks.json')


def main() -> int:
    rows = [json.loads(l) for l in (DATA / 'meta.jsonl').read_text(encoding='utf-8').splitlines()
            if l.strip()]
    items = []
    for r in rows:
        z = np.load(DATA / r['state_path'])
        T = int(z['state'].shape[0])
        for t in range(0, T, 2):
            items.append({'skill': r['skill'], 'episode': int(r['episode_index']),
                          'tick': t, 'frame': t, 'phase': 'overfit_all',
                          'robot_block': 0, 'block_wall': 0})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({'items': items, 'source': str(DATA), 'stride': 2},
                              ensure_ascii=False, indent=2), encoding='utf-8')
    print('picks', len(items), {s: sum(1 for i in items if i['skill'] == s)
                                for s in ('A90', 'B90')}, '->', OUT)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
