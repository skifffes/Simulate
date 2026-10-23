"""实验 2.1 阶段 1：核对实验 2 的数据与评测设置。

在服务器上运行（读取 LeRobot 数据集与实验 2 的评测 JSON）：
    PYTHONPATH=src venv/bin/python -u scripts/audit_experiment_2_1.py \
        --dataset data/lerobot_panda_multi --diag data/_shards_exp2 \
        --results results/experiment_2 --out results/experiment_2_1/audit

输出：
    summary.json   所有检查项的数值结果
    checks.csv     逐条检查（scope, check, ok, detail）
    start_joints.csv  每个 episode 的起始 7 关节角（按任务）
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import statistics as st
from pathlib import Path

import numpy as np

HOME = [0, -0.5, 0, -2.1, 0, 1.65, 0.78]
SKILL_OF_TASK = {
    'pick up the orange block and place it in the target area, keeping its long side aligned with Y': 'A0',
    'pick up the orange block and place it in the target area, with its long side aligned with X': 'A90',
    'push the orange block along the positive X direction to the wall': 'B',
    'pick up the orange block and place it into the tray, with its long side aligned with Y': 'C',
}


def read_meta(root: Path):
    info = json.loads((root / 'meta/info.json').read_text(encoding='utf-8'))
    episodes = [json.loads(line) for line in
                (root / 'meta/episodes.jsonl').read_text(encoding='utf-8').splitlines() if line]
    tasks = [json.loads(line) for line in
             (root / 'meta/tasks.jsonl').read_text(encoding='utf-8').splitlines() if line]
    return info, episodes, tasks


def episode_start_states(root: Path, info, episodes, tasks):
    import pyarrow.parquet as pq

    by_task = {t['task_index']: t['task'] for t in tasks}
    rows = []
    for ep in episodes:
        idx = int(ep['episode_index'])
        chunk = idx // info['chunks_size']
        path = root / info['data_path'].format(episode_chunk=chunk, episode_index=idx)
        table = pq.read_table(path, columns=['observation.state', 'action'])
        state0 = np.asarray(table['observation.state'][0].as_py(), dtype=float)
        action0 = np.asarray(table['action'][0].as_py(), dtype=float)
        rows.append({
            'episode': idx,
            'task': ep['tasks'][0],
            'q0': state0[:7].tolist(),
            'grip0': state0[7:9].tolist(),
            'action0_grip': float(action0[7]),
        })
    return rows


def joint_stats(rows):
    out = {}
    for skill in ('A0', 'A90', 'B', 'C'):
        group = [r for r in rows if SKILL_OF_TASK.get(r['task']) == skill]
        if not group:
            continue
        arr = np.asarray([r['q0'] for r in group])
        out[skill] = {
            'n': len(group),
            'mean': arr.mean(axis=0).round(4).tolist(),
            'std': arr.std(axis=0).round(6).tolist(),
            'min': arr.min(axis=0).round(4).tolist(),
            'max': arr.max(axis=0).round(4).tolist(),
            'unique_rows': len({tuple(np.round(r['q0'], 5)) for r in group}),
            'max_abs_dev_from_home': float(np.abs(arr - np.array(HOME)).max()),
            'grip0_mean': float(np.mean([np.mean(r['grip0']) for r in group])),
        }
    return out


def diag_scenes(diag_root: Path):
    out = {}
    for skill in ('A0', 'A90', 'B', 'C'):
        path = diag_root / f'_diag_{skill}' / 'episodes.jsonl'
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
        ok = [r for r in rows if r.get('success')]
        out[skill] = {
            'attempts': len(rows),
            'success': len(ok),
            'x': [r['scene']['block_start'][0] for r in ok],
            'y': [r['scene']['block_start'][1] for r in ok],
            'yaw': [r['scene']['block_yaw'] for r in ok],
        }
    return out


def eval_lifts(results: Path):
    out = {}
    for skill in ('A0', 'A90', 'B', 'C'):
        p = results / 'eval_atomic' / f'eval_{skill}.json'
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding='utf-8'))
        ok = [e for e in d['episodes'] if e['success']]
        out[skill] = {
            'success': len(ok), 'n': len(d['episodes']),
            'lift_min': min(e['lift_height_m'] for e in ok) if ok else None,
            'lift_median': st.median([e['lift_height_m'] for e in ok]) if ok else None,
            'no_lift_successes': sum(1 for e in ok if e['lift_height_m'] < 0.03),
            'released_all': all(e.get('released') for e in ok) if ok else None,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='data/lerobot_panda_multi')
    ap.add_argument('--diag', default='data/_shards_exp2')
    ap.add_argument('--results', default='results/experiment_2')
    ap.add_argument('--out', default='results/experiment_2_1/audit')
    args = ap.parse_args()

    root = Path(args.dataset)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    info, episodes, tasks = read_meta(root)
    rows = episode_start_states(root, info, episodes, tasks)
    stats = joint_stats(rows)
    scenes = diag_scenes(Path(args.diag))
    lifts = eval_lifts(Path(args.results))

    checks = []

    def check(scope, name, ok, detail=''):
        checks.append({'scope': scope, 'check': name, 'ok': bool(ok), 'detail': str(detail)})

    check('dataset', 'episodes == 480', info['total_episodes'] == 480, info['total_episodes'])
    check('dataset', 'tasks == 4', info['total_tasks'] == 4, info['total_tasks'])
    for skill in ('A0', 'A90', 'B', 'C'):
        s = stats.get(skill, {})
        check('start-states', f'{skill} 起始关节唯一值数 == 1',
              s.get('unique_rows') == 1, s.get('unique_rows'))
        check('start-states', f'{skill} 起始关节与 HOME 最大偏差 < 1e-4',
              s.get('max_abs_dev_from_home', 9) < 1e-4, s.get('max_abs_dev_from_home'))
    for skill in ('B', 'C'):
        sc = scenes.get(skill, {})

        def near_axis(y):
            # 长方体长轴只允许 0 或 90°，按 mod 90° 的圆距离判断。
            d = (y + math.pi / 4) % (math.pi / 2) - math.pi / 4
            return abs(d) < 0.25

        check('object-dist', f'{skill} 物体 yaw 只取 0/90 附近',
              sc.get('yaw') is not None and all(near_axis(y) for y in sc['yaw']),
              sorted({round(math.degrees(y), 0) for y in sc.get('yaw', [])})[:8])
    for skill, v in lifts.items():
        if skill in ('A0', 'A90', 'C'):
            check('eval', f'{skill} 成功集全部有 >3cm 抬升', v['no_lift_successes'] == 0,
                  f"no_lift={v['no_lift_successes']}")

    summary = {
        'dataset': {'episodes': info['total_episodes'], 'frames': info['total_frames'],
                    'tasks': info['total_tasks']},
        'start_joint_stats': stats,
        'object_distributions': {
            k: {'attempts': v['attempts'], 'success': v['success'],
                'x_range': [min(v['x']), max(v['x'])],
                'y_range': [min(v['y']), max(v['y'])],
                'yaw_values': sorted({round(y, 3) for y in v['yaw']})}
            for k, v in scenes.items()
        },
        'eval_lift': lifts,
        'checks': checks,
        'failures': [c for c in checks if not c['ok']],
    }
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                      encoding='utf-8')
    with (out / 'checks.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=['scope', 'check', 'ok', 'detail'])
        w.writeheader()
        w.writerows(checks)
    with (out / 'start_joints.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['episode', 'skill'] + [f'q{i + 1}' for i in range(7)] + ['grip_mean'])
        for r in rows:
            w.writerow([r['episode'], SKILL_OF_TASK.get(r['task'], '?')]
                       + [round(v, 5) for v in r['q0']] + [round(float(np.mean(r['grip0'])), 5)])

    print(json.dumps({'failures': summary['failures'],
                      'start_stats': {k: {'n': v['n'], 'unique_rows': v['unique_rows']}
                                      for k, v in stats.items()}}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
