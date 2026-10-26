"""实验 2.3 阶段 1：多起点生成器的合法性与覆盖探测（不跑专家）。

对三种采样模式各抽 N 个候选起点，逐个通过 starts.validate_start 的检查，
统计接受率、拒绝原因、起始构型到 HOME 的距离、夹爪-物体间距、物体 yaw 等。
负对照（故意非法构型）在 smoke 文档中单独记录。

用法（服务器）：
    PYTHONPATH=src venv/bin/python -u scripts/probe_start_sampler.py \
        --per-mode 100 --out results/experiment_2_3/starts
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pybullet as p

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from simulation import HOME  # noqa: E402
from vla.atomic_tasks.config2 import CAMERAS, IMAGE_SIZE  # noqa: E402
from vla.atomic_tasks.scene2 import PLACE, build_scene  # noqa: E402
from vla.atomic_tasks.starts import SAMPLERS, validate_start  # noqa: E402
from vla.cameras import CameraRig  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-mode', type=int, default=100)
    ap.add_argument('--seed-base', type=int, default=51000000)
    ap.add_argument('--modes', nargs='+',
                    default=['home', 'home_neighborhood', 'multi_ik', 'task_relevant'])
    ap.add_argument('--out', type=Path, default=ROOT / 'results/experiment_2_3/starts')
    ap.add_argument('--scene-every', type=int, default=20)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed_base)
    sim = block = rig = None
    rows = []
    for i in range(args.per_mode * len(args.modes)):
        mode = args.modes[i % len(args.modes)]
        if i % args.scene_every == 0:
            if sim is not None:
                p.disconnect(sim.client)
            yaw = float(rng.choice([0.0, math.pi / 2]) + rng.uniform(-0.10, 0.10))
            sim, block, wall, tray, info = build_scene(seed=args.seed_base + i,
                                                       start=PLACE, jitter=0.012, yaw=yaw)
            rig = CameraRig(sim, CAMERAS, IMAGE_SIZE, supersample=1)
        block_pos, _ = p.getBasePositionAndOrientation(block, physicsClientId=sim.client)
        block_pos = np.asarray(block_pos, dtype=float)
        cand = SAMPLERS[mode](rng, sim, block_pos, 0.0, 0)
        checks = validate_start(sim, block, cand.q, rig, block_pos=block_pos)
        rows.append({
            'mode': mode, 'candidate': i, 'accepted': bool(checks['accepted']),
            'reject': checks.get('reject'),
            'home_distance_rad': round(float(np.linalg.norm(np.asarray(cand.q) - np.asarray(HOME))), 4),
            'min_limit_margin': checks.get('min_limit_margin'),
            'block_clearance_m': checks.get('block_clearance_m'),
            'top_visible': (checks.get('camera') or {}).get('top', {}).get('inside'),
            'wrist_in_frustum': (checks.get('camera') or {}).get('wrist', {}).get('inside'),
            'ik_error_m': checks.get('b_approach_ik_error_m'),
            'block_yaw_deg': checks.get('block_yaw_deg'),
            'block_x': round(float(block_pos[0]), 4),
            'block_y': round(float(block_pos[1]), 4),
            'q': [round(float(v), 4) for v in cand.q],
            'ee_target': None if cand.ee_target is None else [round(float(v), 4) for v in cand.ee_target],
            'gripper_yaw': round(float(cand.gripper_yaw), 3),
        })
    if sim is not None:
        p.disconnect(sim.client)

    with (args.out / 'candidates.jsonl').open('w', encoding='utf-8') as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + '\n')
    with (args.out / 'candidates.csv').open('w', newline='', encoding='utf-8-sig') as fh:
        writer = csv.DictWriter(fh, fieldnames=[k for k in rows[0] if k != 'q'],
                                extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for mode in args.modes:
        sub = [r for r in rows if r['mode'] == mode]
        accepted = [r for r in sub if r['accepted']]
        rejects = {}
        for r in sub:
            if not r['accepted']:
                rejects[r['reject']] = rejects.get(r['reject'], 0) + 1
        dist = [r['home_distance_rad'] for r in accepted]
        summary[mode] = {
            'n': len(sub), 'accepted': len(accepted),
            'accept_rate': round(len(accepted) / len(sub), 4),
            'reject_reasons': rejects,
            'home_distance_rad': {
                'mean': round(float(np.mean(dist)), 3) if dist else None,
                'min': round(float(np.min(dist)), 3) if dist else None,
                'max': round(float(np.max(dist)), 3) if dist else None,
            },
            'block_clearance_m_mean': round(float(np.mean(
                [r['block_clearance_m'] for r in accepted])), 3) if accepted else None,
            'top_visible_rate': round(float(np.mean([bool(r['top_visible']) for r in accepted])), 3) if accepted else None,
            'wrist_in_frustum_rate': round(float(np.mean([bool(r['wrist_in_frustum']) for r in accepted])), 3) if accepted else None,
            'block_yaw_deg': sorted({round(float(r['block_yaw_deg']), 1) for r in accepted})[:8],
        }
    payload = {'per_mode': args.per_mode, 'seed_base': args.seed_base,
               'scene_every': args.scene_every, 'modes': summary}
    (args.out / 'probe.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                         encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
