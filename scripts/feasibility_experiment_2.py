"""实验 2 阶段 1：四条组合链的专家物理可行性检查。

对 A0→B / A90→B / A0→C / A90→C 各跑一次专家链：A 成功后不重置任何状态，
B/C 直接从 A 的真实终态继续。输出 JSON、俯视截图和逐阶段日志。

用法：
    python scripts/feasibility_experiment_2.py --out results/experiment_2/feasibility
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from vla.atomic_tasks.experts import SkillExpert  # noqa: E402
from vla.atomic_tasks.scene2 import (  # noqa: E402
    PLACE, TRAY, WALL_INNER_X, build_scene,
)
from vla.cameras import CameraRig  # noqa: E402

CHAINS = [('A0', 'B'), ('A90', 'B'), ('A0', 'C'), ('A90', 'C')]


def run_chain(a_kind: str, b_kind: str, record_dir: Path, seed: int = 0,
              jitter: float = 0.0, tag_suffix: str = ''):
    sim, block, wall, tray, info = build_scene(seed=seed, jitter=jitter)
    cam = CameraRig(sim, supersample=1)
    started = time.perf_counter()
    try:
        a_res = SkillExpert(sim, block)
        a = a_res.skill_a0() if a_kind == 'A0' else a_res.skill_a90()
        state_after_a = info_after(sim, block)
        b = {'skill': b_kind, 'skipped': True, 'reason': 'A failed'}
        if a['success']:
            b_expert = SkillExpert(sim, block)
            b = b_expert.skill_b() if b_kind == 'B' else b_expert.skill_c()
        shot = Image.fromarray(np.asarray(cam.capture()['top'], dtype=np.uint8))
        tag = f'{a_kind}_{b_kind}{tag_suffix}'
        record_dir.mkdir(parents=True, exist_ok=True)
        shot.save(record_dir / f'{tag}_final.png')
        return {
            'chain': f'{a_kind}_{b_kind}',
            'seed': seed,
            'scene': info,
            'A': a,
            'B_or_C': b,
            'state_after_A': state_after_a,
            'final': info_after(sim, block),
            'wall_inner_x': WALL_INNER_X,
            'tray': list(TRAY),
            'wall_seconds': round(time.perf_counter() - started, 2),
        }
    finally:
        sim.close()


def info_after(sim, block):
    import pybullet as p
    pos, quat = p.getBasePositionAndOrientation(block, physicsClientId=sim.client)
    euler = p.getEulerFromQuaternion(quat)
    return {
        'position': [float(v) for v in pos],
        'yaw_rad': float(np.arctan2(np.sin(euler[2]), np.cos(euler[2]))),
        'joints': [float(v) for v in sim.q()],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/experiment_2/feasibility')
    ap.add_argument('--chain', help='只跑一条链，例如 A0_B')
    ap.add_argument('--trials', type=int, default=1, help='每条链重复次数（不同种子）')
    ap.add_argument('--seed-base', type=int, default=7000)
    ap.add_argument('--jitter', type=float, default=0.006, help='物体初始位置扰动（m）')
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    chains = CHAINS
    if args.chain:
        a, b = args.chain.split('_')
        chains = [(a, b)]

    results = []
    for a_kind, b_kind in chains:
        chain_ok = 0
        for t in range(args.trials):
            seed = args.seed_base + t
            res = run_chain(a_kind, b_kind, out, seed=seed, jitter=args.jitter,
                            tag_suffix=f'_s{seed}')
            results.append(res)
            ok = bool(res['A']['success'] and res['B_or_C'].get('success'))
            chain_ok += int(ok)
            print(f"[{res['chain']} s{seed}] A={res['A']['success']} "
                  f"({res['A']['reason']}) B/C={res['B_or_C'].get('success')} "
                  f"({res['B_or_C'].get('reason')}) "
                  f"A_yaw={res['A'].get('final', {}).get('yaw_rad', float('nan')):.2f}",
                  flush=True)
        print(f'== {a_kind}_{b_kind}: {chain_ok}/{args.trials} full-chain success', flush=True)

    (out / 'feasibility.json').write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    summary = {'trials': args.trials, 'jitter': args.jitter, 'chains': []}
    for a_kind, b_kind in chains:
        rows = [r for r in results if r['chain'] == f'{a_kind}_{b_kind}']
        a_ok = sum(1 for r in rows if r['A']['success'])
        b_ok = sum(1 for r in rows if r['B_or_C'].get('success'))
        full = sum(1 for r in rows if r['A']['success'] and r['B_or_C'].get('success'))
        summary['chains'].append({
            'chain': f'{a_kind}_{b_kind}', 'trials': len(rows),
            'A_success': a_ok, 'B_or_C_success': b_ok, 'full_chain_success': full,
            'A_reasons': sorted({r['A']['reason'] for r in rows}),
            'B_reasons': sorted({str(r['B_or_C'].get('reason')) for r in rows}),
        })
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                      encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
