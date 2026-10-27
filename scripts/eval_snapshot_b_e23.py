"""实验 2.3：固定交接状态对照——把 B 策略放到 2.1 的真实 A 终态上执行。

这些快照由**原始 A 模型**产生，作为"相同交接状态"的诊断起点；快照恢复
只用于诊断，不计入真实连续组合。B 仍用统一 v2.2 判据。

用法（服务器）：
    HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src venv/bin/python \
        scripts/eval_snapshot_b_e23.py --a-skill A90 --episodes 20 --tag D2 \
        --policy outputs/smolvla_e23_D2/checkpoints/020000/pretrained_model \
        --out results/experiment_2_3/eval/snapshot/D2
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pybullet as p

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from eval_handoff_timing import (  # noqa: E402
    final_ok, finger_contacts, obj_state, task_ok,
)
from eval_multi_atomic_vla import PolicyRunner  # noqa: E402
from manipulation import finger_widths, set_gripper  # noqa: E402
from vla.atomic_tasks.config2 import CAMERAS, IMAGE_SIZE, STEPS_PER_ACTION, TASKS  # noqa: E402
from vla.atomic_tasks.scene2 import build_scene  # noqa: E402
from vla.cameras import CameraRig  # noqa: E402


def run(runner, snap: dict, bullet: Path, args):
    sim, block, wall, tray, info = build_scene(seed=snap['seed'], jitter=0.008)
    try:
        p.restoreState(fileName=str(bullet), physicsClientId=sim.client)
        sim.command(np.asarray(snap['last_action'][:7], dtype=float))
        set_gripper(sim, 0.04, 20.0, 0.06)
        for _ in range(5):
            sim.step()
        camera = CameraRig(sim, CAMERAS, IMAGE_SIZE, supersample=1)
        st0 = obj_state(sim, block)
        y0, yaw0 = float(st0['position'][1]), st0['yaw']
        runner.reset()
        np.random.seed(snap['seed'])
        runner.torch.manual_seed(snap['seed'])
        runner.torch.cuda.manual_seed_all(snap['seed'])
        started = time.perf_counter()
        b_ok, first_success = False, None
        tick = 0
        last_target = np.asarray(sim.q(), dtype=float)
        last_width = 0.04
        for tick in range(args.max_ticks_b):
            frames = camera.capture()
            state = np.concatenate([sim.q(), np.asarray(finger_widths(sim))])
            action = runner.act(frames, state, TASKS['B'])
            target = np.clip(action[:7], np.asarray(sim.lower) + 1e-4,
                             np.asarray(sim.upper) - 1e-4)
            width = float(np.clip(action[7], 0.0, 0.04))
            last_target, last_width = target, width
            sim.command(target)
            set_gripper(sim, width, 20.0, 0.05)
            for _ in range(STEPS_PER_ACTION):
                sim.step()
            st = obj_state(sim, block)
            if tick > 15 and task_ok('B', st, y0, yaw0):
                b_ok = True
                first_success = tick
                break
        for _ in range(int(round(1.0 / sim.dt))):
            sim.command(last_target)
            set_gripper(sim, last_width, 20.0, 0.05)
            sim.step()
        stable = True
        for _ in range(5):
            if obj_state(sim, block)['speed'] >= 0.01:
                stable = False
            for _ in range(int(round(0.04 / sim.dt))):
                sim.step()
        final = obj_state(sim, block)
        released = not finger_contacts(sim, block)
        good = bool(b_ok and final_ok('B', final, released, stable, y0, yaw0))
        return {
            'policy_tag': args.tag, 'a_skill': args.a_skill, 'seed': snap['seed'],
            'success': good, 'failure': None if good else ('timeout' if first_success is None
                                                           else 'final_check'),
            'first_success_tick': first_success, 'ticks': int(tick + 1),
            'wall_seconds': round(time.perf_counter() - started, 2),
            'released': bool(released), 'stable': bool(stable),
            'switch_object': {'position': st0['position'].tolist(), 'yaw': float(yaw0)},
            'final_object': {'position': final['position'].tolist(),
                             'yaw': float(final['yaw']), 'tilt_deg': float(final['tilt'])},
        }
    finally:
        if p.isConnected(sim.client):
            p.disconnect(sim.client)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', required=True)
    ap.add_argument('--a-skill', choices=['A0', 'A90'], required=True)
    ap.add_argument('--episodes', type=int, default=20)
    ap.add_argument('--snapshots', type=Path, default=ROOT / 'results/experiment_2_1/snapshots')
    ap.add_argument('--max-ticks-b', type=int, default=260)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()

    snaps = []
    for path in sorted(args.snapshots.glob(f'{args.a_skill}_*.json')):
        if 'restore' in path.name or 'collection' in path.name:
            continue
        snaps.append(json.loads(path.read_text(encoding='utf-8')))
    snaps = snaps[:args.episodes]
    args.out.mkdir(parents=True, exist_ok=True)
    runner = PolicyRunner(args.policy)
    rows = []
    for i, snap in enumerate(snaps):
        bullet = args.snapshots / snap['bullet_file']
        rec = run(runner, snap, bullet, args)
        rows.append(rec)
        print(f'[{args.tag} snapshot {args.a_skill} {i + 1}/{len(snaps)}] '
              f'seed={snap["seed"]} {"OK" if rec["success"] else "FAIL"} '
              f'({rec["failure"]}) ticks={rec["ticks"]}', flush=True)
    n = len(rows)
    summary = {
        'tag': args.tag, 'a_skill': args.a_skill, 'n': n,
        'success': sum(1 for r in rows if r['success']),
        'rate': round(sum(1 for r in rows if r['success']) / max(n, 1), 4),
        'ticks_mean': float(np.mean([r['ticks'] for r in rows])),
    }
    (args.out / f'snapshot_{args.a_skill}.json').write_text(
        json.dumps({'summary': summary, 'episodes': rows}, ensure_ascii=False, indent=2),
        encoding='utf-8')
    fields = ['policy_tag', 'a_skill', 'seed', 'success', 'failure', 'ticks',
              'first_success_tick', 'released', 'stable']
    with (args.out / f'snapshot_{args.a_skill}.csv').open('w', newline='',
                                                          encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
