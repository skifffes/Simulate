"""实验 2.3：A0/A90 → B 的**真实连续**直接交接评测（统一 v2.2）。

流程：
1. 从新的测试种子建场景，运行 A（A0 或 A90），直到 v2.2 的 A 成功条件满足；
2. **立即**切换语言条件到 B、`policy.reset()` 清空旧动作块；
3. 不重置机器人/物体、不回 HOME，由同一个 VLA 继续执行 B；
4. B 用 v2.2 判定（任务条件 → 停发动作 → 静置 1.0 s → 复核）。

用法（服务器）：
    HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src venv/bin/python \
        scripts/eval_composition_e23.py \
        --policy outputs/smolvla_e23_D1/checkpoints/020000/pretrained_model \
        --chain A0_B --episodes 20 --tag D1 \
        --out results/experiment_2_3/eval/composition/D1 --seed-base 40000000
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
    a_success, final_ok, finger_contacts, obj_state, task_ok,
)
from eval_multi_atomic_vla import PolicyRunner  # noqa: E402
from manipulation import finger_widths, set_gripper  # noqa: E402
from vla.atomic_tasks.config2 import CAMERAS, IMAGE_SIZE, STEPS_PER_ACTION, TASKS  # noqa: E402
from vla.atomic_tasks.scene2 import build_scene  # noqa: E402
from vla.cameras import CameraRig  # noqa: E402


def run_chain(runner, chain: str, seed: int, args):
    a_skill, b_skill = chain.split('_')
    sim, block, wall, tray, info = build_scene(seed=seed, jitter=0.008, yaw=0.0)
    try:
        camera = CameraRig(sim, CAMERAS, IMAGE_SIZE, supersample=1)
        for _ in range(args.warmup_steps):
            sim.step()
        pos0, _ = p.getBasePositionAndOrientation(block, physicsClientId=sim.client)
        initial_z = float(pos0[2])
        max_z = initial_z
        runner.reset()
        np.random.seed(seed)
        runner.torch.manual_seed(seed)
        runner.torch.cuda.manual_seed_all(seed)

        a_ok = False
        a_ticks = 0
        a_last = np.asarray(sim.q(), dtype=float)
        a_width = 0.04
        started = time.perf_counter()
        for tick in range(args.max_ticks_a):
            frames = camera.capture()
            state = np.concatenate([sim.q(), np.asarray(finger_widths(sim))])
            action = runner.act(frames, state, TASKS[a_skill])
            target = np.clip(action[:7], np.asarray(sim.lower) + 1e-4,
                             np.asarray(sim.upper) - 1e-4)
            width = float(np.clip(action[7], 0.0, 0.04))
            a_last, a_width = target, width
            sim.command(target)
            set_gripper(sim, width, 20.0, 0.05)
            for _ in range(STEPS_PER_ACTION):
                sim.step()
            pos, _ = p.getBasePositionAndOrientation(block, physicsClientId=sim.client)
            max_z = max(max_z, float(pos[2]))
            a_ticks = tick + 1
            if tick > 20 and a_success(a_skill, sim, block, initial_z, max_z):
                a_ok = True
                break
        st = obj_state(sim, block)
        switch = {'position': st['position'].tolist(), 'yaw': float(st['yaw']),
                  'robot_q': [float(v) for v in sim.q()]}
        if not a_ok:
            return {
                'policy_tag': args.tag, 'chain': chain, 'seed': seed,
                'A_success': False, 'B_success': False, 'success': False,
                'failure': 'A_failed', 'a_ticks': a_ticks, 'b_ticks': 0,
                'wall_seconds': round(time.perf_counter() - started, 2),
                'switch_object': switch,
            }

        # 真实连续交接：立即切换语言、清空旧动作块；不重置任何状态
        runner.reset()
        st = obj_state(sim, block)
        y0, yaw0 = float(st['position'][1]), st['yaw']
        b_ok = False
        first_success = None
        b_tick = 0
        last_target = np.asarray(sim.q(), dtype=float)
        last_width = 0.04
        for b_tick in range(args.max_ticks_b):
            frames = camera.capture()
            state = np.concatenate([sim.q(), np.asarray(finger_widths(sim))])
            action = runner.act(frames, state, TASKS[b_skill])
            target = np.clip(action[:7], np.asarray(sim.lower) + 1e-4,
                             np.asarray(sim.upper) - 1e-4)
            width = float(np.clip(action[7], 0.0, 0.04))
            last_target, last_width = target, width
            sim.command(target)
            set_gripper(sim, width, 20.0, 0.05)
            for _ in range(STEPS_PER_ACTION):
                sim.step()
            st2 = obj_state(sim, block)
            if b_tick > 15 and task_ok(b_skill, st2, y0, yaw0):
                b_ok = True
                first_success = b_tick
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
        good = bool(a_ok and b_ok and final_ok(b_skill, final, released, stable, y0, yaw0))
        return {
            'policy_tag': args.tag, 'chain': chain, 'seed': seed,
            'A_success': True, 'B_success': good, 'success': good,
            'failure': None if good else ('timeout' if first_success is None else 'final_check'),
            'a_ticks': a_ticks, 'b_ticks': int(b_tick + 1),
            'first_success_tick': first_success,
            'wall_seconds': round(time.perf_counter() - started, 2),
            'released': bool(released), 'stable': bool(stable),
            'final_object': {'position': final['position'].tolist(),
                             'yaw': float(final['yaw']), 'tilt_deg': float(final['tilt'])},
            'switch_object': switch,
        }
    finally:
        if p.isConnected(sim.client):
            p.disconnect(sim.client)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', required=True)
    ap.add_argument('--chain', choices=['A0_B', 'A90_B'], required=True)
    ap.add_argument('--episodes', type=int, default=20)
    ap.add_argument('--seed-base', type=int, default=40000000)
    ap.add_argument('--max-ticks-a', type=int, default=260)
    ap.add_argument('--max-ticks-b', type=int, default=260)
    ap.add_argument('--warmup-steps', type=int, default=30)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    runner = PolicyRunner(args.policy)
    rows = []
    for i in range(args.episodes):
        seed = args.seed_base + i
        rec = run_chain(runner, args.chain, seed, args)
        rows.append(rec)
        print(f'[{args.tag} {args.chain} {i + 1}/{args.episodes}] seed={seed} '
              f'A={rec["A_success"]} B={rec["B_success"]} fail={rec["failure"]} '
              f'ticks={rec["a_ticks"]}+{rec["b_ticks"]}', flush=True)

    n = len(rows)
    summary = {
        'tag': args.tag, 'chain': args.chain, 'n': n,
        'A_success': sum(1 for r in rows if r['A_success']),
        'B_success': sum(1 for r in rows if r['B_success']),
        'chain_success': sum(1 for r in rows if r['success']),
        'chain_rate': round(sum(1 for r in rows if r['success']) / max(n, 1), 4),
        'conditional_B_success': sum(1 for r in rows if r['A_success'] and r['B_success']),
        'a_ticks_mean': float(np.mean([r['a_ticks'] for r in rows])),
    }
    (args.out / f'{args.chain}.json').write_text(
        json.dumps({'summary': summary, 'episodes': rows}, ensure_ascii=False, indent=2),
        encoding='utf-8')
    fields = ['policy_tag', 'chain', 'seed', 'A_success', 'B_success', 'success',
              'failure', 'a_ticks', 'b_ticks', 'released', 'stable']
    with (args.out / f'{args.chain}.csv').open('w', newline='', encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
