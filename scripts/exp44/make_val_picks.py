"""实验 4.4：构造 val 评测用的状态样本（固定、可复现）。

从 `/scratch/lijh/exp43/export_v3/meta.jsonl` 的 val 行（16 条，冻结）中，
按技能分层采样固定数量的状态（默认 A90 15 + B90 15 = 30），每条间隔采样。
输出 picks json（供 expert_state_query.py 使用）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DATA = Path('/scratch/lijh/exp43/export_v3')
OUT = Path('/scratch/lijh/exp40/repo/results/experiment_4_4/val_picks.json')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-skill', type=int, default=15)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=str(OUT))
    args = ap.parse_args()
    meta = [json.loads(l) for l in (DATA / 'meta.jsonl').read_text(encoding='utf-8').splitlines()
            if l.strip()]
    val = [m for m in meta if m.get('split') == 'val']
    rng = np.random.default_rng(args.seed)
    items = []
    for skill in ('A90', 'B90'):
        eps = [m for m in val if m['skill'] == skill]
        if not eps:
            continue
        picked = []
        i = 0
        while len(picked) < args.per_skill and eps:
            e = eps[i % len(eps)]
            z = np.load(DATA / e['state_path'])
            T = int(z['state'].shape[0])
            t = int(rng.integers(0, T))
            picked.append({'skill': skill, 'episode': int(e['episode_index']),
                           'tick': t, 'frame': t, 'phase': 'val_sample',
                           'robot_block': 0, 'block_wall': 0})
            i += 1
            if i > 200:
                break
        items.extend(picked)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({'items': items, 'source': str(DATA),
                                          'split': 'val', 'seed': args.seed},
                                         ensure_ascii=False, indent=2), encoding='utf-8')
    print('val picks:', len(items), {s: sum(1 for i in items if i['skill'] == s)
                                     for s in ('A90', 'B90')}, '->', args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
