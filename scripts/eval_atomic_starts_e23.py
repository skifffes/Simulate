"""实验 2.3：冻结 VLA 从**未知合法起点**执行 B 的评测（统一 v2.2 判据）。

起点来自 results/experiment_2_3/eval/holdout_starts.json（种子 81 000 000+，
与训练 70 000 000+、测试 40 000 000+ 隔离）。每个起点只在 episode 开始时
初始化机器人构型；执行中不回 HOME、不重置。

用法（服务器）：
    HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=4 PYTHONPATH=src venv/bin/python \
        scripts/eval_atomic_starts_e23.py \
        --policy outputs/smolvla_e23_D2/checkpoints/020000/pretrained_model \
        --starts results/experiment_2_3/eval/holdout_starts.json \
        --tag D2 --out results/experiment_2_3/eval/atomic_starts/D2
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
    finger_contacts, final_ok, obj_state, task_ok,
)
from eval_multi_atomic_vla import PolicyRunner  # noqa: E402
from manipulation import finger_widths, set_gripper  # noqa: E402
from vla.atomic_tasks.config2 import CAMERAS, IMAGE_SIZE, STEPS_PER_ACTION, TASKS  # noqa: E402
from vla.atomic_tasks.scene2 import build_scene  # noqa: E402
from vla.atomic_tasks.starts import set_robot_state  # noqa: E402
from vla.cameras import CameraRig  # noqa: E402


def run(runner, start: dict, args):
    sim, block, wall, tray, info = build_scene(
        seed=start['seed'], start=tuple(start['block_start']), jitter=0.0,
        yaw=start['block_yaw_rad'])
    try:
        camera = CameraRig(sim, CAMERAS, IMAGE_SIZE, supersample=1)
        set_robot_state(sim, np.asarray(start['q'], dtype=float))
        sim.command(np.asarray(start['q'], dtype=float))
        set_gripper(sim, 0.035, 20.0, 0.05)
        for _ in range(20):
            sim.step()
        st0 = obj_state(sim, block)
        y0, yaw0 = float(st0['position'][1]), st0['yaw']
        start_q = sim.q().copy()

        runner.reset()
        np.random.seed(start['seed'])
        runner.torch.manual_seed(start['seed'])
        runner.torch.cuda.manual_seed_all(start['seed'])
        started = time.perf_counter()
        success, first_success = False, None
        tick = 0
        last_target = np.asarray(sim.q(), dtype=float)
        last_width = 0.035
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
                success = True
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
        good = bool(success and final_ok('B', final, released, stable, y0, yaw0))
        return {
            'group': start['group'], 'mode': start['mode'], 'seed': start['seed'],
            'start_q': [round(float(v), 5) for v in start_q],
            'home_distance_rad': start['home_distance_rad'],
            'success': good, 'failure': None if good else ('timeout' if first_success is None
                                                           else 'final_check'),
            'first_success_tick': first_success, 'ticks': int(tick + 1),
            'wall_seconds': round(time.perf_counter() - started, 2),
            'released': bool(released), 'stable': bool(stable),
            'final_object': {'position': final['position'].tolist(),
                             'yaw': float(final['yaw']), 'tilt_deg': float(final['tilt'])},
            'switch_object': {'position': st0['position'].tolist(), 'yaw': float(yaw0)},
        }
    finally:
        if p.isConnected(sim.client):
            p.disconnect(sim.client)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', required=True)
    ap.add_argument('--starts', type=Path, required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--episodes', type=int, default=0, help='0 = 全部起点')
    ap.add_argument('--max-ticks-b', type=int, default=260)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()

    starts = json.loads(args.starts.read_text(encoding='utf-8'))
    if args.episodes:
        starts = starts[:args.episodes]
    args.out.mkdir(parents=True, exist_ok=True)
    runner = PolicyRunner(args.policy)
    rows = []
    for i, start in enumerate(starts):
        rec = run(runner, start, args)
        rec['policy_tag'] = args.tag
        rows.append(rec)
        print(f'[{args.tag} {i + 1}/{len(starts)}] {start["group"]} {start["mode"]} '
              f'seed={start["seed"]} {"OK" if rec["success"] else "FAIL"} '
              f'({rec["failure"]}) ticks={rec["ticks"]}', flush=True)

    def rate(sub):
        k = sum(1 for r in sub if r['success'])
        return {'n': len(sub), 'success': k,
                'rate': round(k / len(sub), 4) if sub else 0.0}

    summary = {
        'tag': args.tag, 'policy': args.policy, 'n': len(rows),
        'all': rate(rows),
        'by_yaw': {g: rate([r for r in rows if r['group'] == g])
                   for g in ('yaw0', 'yaw90')},
        'by_mode': {m: rate([r for r in rows if r['mode'] == m])
                    for m in ('home_neighborhood', 'multi_ik', 'task_relevant')},
    }
    (args.out / 'eval.json').write_text(
        json.dumps({'summary': summary, 'episodes': rows}, ensure_ascii=False, indent=2),
        encoding='utf-8')
    fields = ['policy_tag', 'group', 'mode', 'seed', 'success', 'failure', 'ticks',
              'home_distance_rad', 'released', 'stable']
    with (args.out / 'episodes.csv').open('w', newline='', encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
