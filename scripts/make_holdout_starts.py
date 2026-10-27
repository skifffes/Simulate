"""实验 2.3：生成评测用的未知合法起点（与训练起点种子隔离）。

每个 yaw 组 20 个起点：HOME 邻域 10、多 IK 5、任务相关 5。
起点种子从 81 000 000 起，和训练（70 000 000+）、测试（40 000 000+）隔离。

用法（服务器）：
    HF_HUB_OFFLINE=1 PYTHONPATH=src venv/bin/python \
        scripts/make_holdout_starts.py --out results/experiment_2_3/eval/holdout_starts.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from vla.atomic_tasks.config2 import CAMERAS, IMAGE_SIZE  # noqa: E402
from vla.atomic_tasks.scene2 import PLACE, build_scene  # noqa: E402
from vla.atomic_tasks.starts import sample_valid_start  # noqa: E402
from vla.cameras import CameraRig  # noqa: E402

PER_GROUP = 20
MODES = (['home_neighborhood'] * 10 + ['multi_ik'] * 5 + ['task_relevant'] * 5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path,
                    default=ROOT / 'results/experiment_2_3/eval/holdout_starts.json')
    ap.add_argument('--seed-base', type=int, default=81000000)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    starts = []
    for group, base in (('yaw0', 0.0), ('yaw90', math.pi / 2)):
        for k in range(PER_GROUP):
            mode = MODES[k]
            seed = args.seed_base + (0 if group == 'yaw0' else 50000) + k * 7
            rng = np.random.default_rng(seed)
            yaw = base + float(rng.uniform(-0.10, 0.10))
            sim, block, wall, tray, info = build_scene(seed=seed, start=PLACE,
                                                       jitter=0.012, yaw=yaw)
            try:
                camera = CameraRig(sim, CAMERAS, IMAGE_SIZE, supersample=1)
                import pybullet as p
                block_pos, _ = p.getBasePositionAndOrientation(block,
                                                               physicsClientId=sim.client)
                block_pos = np.asarray(block_pos, dtype=float)
                cand, checks, rejections = sample_valid_start(
                    sim, block, camera, mode, rng, block_pos=block_pos)
            finally:
                if p.isConnected(sim.client):
                    p.disconnect(sim.client)
            if cand is None:
                raise SystemExit(f'{group} {mode} 采样失败: {rejections}')
            starts.append({
                'group': group, 'mode': mode, 'seed': seed,
                'q': [round(float(v), 6) for v in cand.q],
                'block_start': [round(float(v), 6) for v in block_pos],
                'block_yaw_rad': round(math.atan2(math.sin(info['block_yaw']),
                                                  math.cos(info['block_yaw'])), 5),
                'home_distance_rad': round(float(np.linalg.norm(
                    np.asarray(cand.q) - np.array([0, -0.5, 0, -2.1, 0, 1.65, 0.78]))), 4),
                'checks': {k: checks.get(k) for k in
                           ('min_limit_margin', 'block_clearance_m', 'b_approach_ik_error_m')},
            })
            print(f'[{group} {k + 1}/{PER_GROUP}] {mode} seed={seed} '
                  f'dist={starts[-1]["home_distance_rad"]:.2f}', flush=True)

    args.out.write_text(json.dumps(starts, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'out': str(args.out), 'n': len(starts),
                      'yaw0': sum(1 for s in starts if s['group'] == 'yaw0'),
                      'yaw90': sum(1 for s in starts if s['group'] == 'yaw90')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
