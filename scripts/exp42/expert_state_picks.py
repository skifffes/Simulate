"""Gate 4 任务 2（第 1 步）：从专家回放 trace 里挑"接近 / 首次接触 / 持续推动"阶段的状态。

纯 NumPy，读归档 CSV + `meta.jsonl`，不跑 PyBullet、不用 GPU。
输出 `expert_state_picks.json`：每条 episode 选中的 tick、阶段、以及对应的数据集帧号。

用法（本地或服务器均可）：
  python scripts/exp42/expert_state_picks.py --trace-dir results/experiment_4_2/contract_replay_p0c
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = Path('/scratch/lijh/exp42/data')


def read_trace(path: Path) -> list[dict]:
    rows = [r for r in csv.DictReader(open(path, encoding='utf-8')) if r['mode'] == 'adapter']
    return rows


def pick(rows: list[dict], skill: str) -> list[dict]:
    n = len(rows)
    touch = [int(r['robot_block_contacts']) > 0 or int(r['block_wall_contacts']) > 0
             for r in rows]
    wall = [int(r['block_wall_contacts']) > 0 for r in rows]
    first = next((i for i, t in enumerate(touch) if t), None)
    out = []

    def add(idx, phase):
        if idx is None or idx < 0 or idx >= n:
            return
        out.append({'tick': int(rows[idx]['tick']), 'phase': phase,
                    'robot_block': int(rows[idx]['robot_block_contacts']),
                    'block_wall': int(rows[idx]['block_wall_contacts']),
                    'frame': int(rows[idx]['tick'])})

    if first is None:
        for i in np.linspace(0, n - 1, 6).astype(int):
            add(int(i), 'approach')
        return out

    # 接近段：首次接触前的 4 个等距 tick（若不足则少取）
    pre = list(range(0, first))
    for i in np.linspace(0, len(pre) - 1, 4).astype(int) if pre else []:
        add(pre[int(i)], 'approach')
    add(first - 1, 'pre_contact')
    add(first, 'contact_onset')

    if skill == 'B90':
        push = [i for i in range(first, n) if wall[i]]
        for i in np.linspace(0, len(push) - 1, 4).astype(int) if push else []:
            add(push[int(i)], 'push')
    else:
        grasp = [i for i in range(first, n) if touch[i]]
        for i in np.linspace(0, len(grasp) - 1, 4).astype(int) if grasp else []:
            add(grasp[int(i)], 'grasp')
        tail = [i for i in range(first, n) if touch[i]]
        if tail:
            add(tail[-1], 'place')
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--trace-dir',
                    default=str(ROOT / 'results' / 'experiment_4_2' / 'contract_replay_p0c'))
    ap.add_argument('--episodes', default='B90:0,1;A90:5')
    ap.add_argument('--out', default=str(ROOT / 'results' / 'experiment_4_2' /
                                         'contract_replay_p0c' / 'expert_state_picks.json'))
    args = ap.parse_args(argv)
    trace_dir = Path(args.trace_dir)
    meta = [json.loads(l) for l in (DATA / 'meta.jsonl').read_text(encoding='utf-8').splitlines()
            if l.strip()]

    result = {'source': str(trace_dir), 'items': []}
    for spec in args.episodes.split(';'):
        skill, eps = spec.split(':')
        for ep in [int(x) for x in eps.split(',')]:
            trace = trace_dir / f'expert_vs_adapter_trace_{skill}_measured.csv'
            rows = [r for r in read_trace(trace) if int(r['episode']) == ep]
            picks = pick(rows, skill)
            for p in picks:
                p.update({'skill': skill, 'episode': ep})
            result['items'].extend(picks)
            print(f'{skill} ep{ep}: {len(picks)} picks '
                  f'({", ".join(sorted({p["phase"] for p in picks}))})')

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('wrote', args.out, 'items', len(result['items']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
