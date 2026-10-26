"""实验 2.2 阶段 2：交接时机对照 H0/H1/H2/H3（统一协议 v2.2）。

流程（每个 A 快照）：
  1. 用冻结 A 策略从同一初始场景重跑 A，找到统一判据下的 t_a_success，
     在该 tick 保存分支快照（.bullet）与 A 的剩余动作队列。
  2. 对每个 H 条件：从分支快照恢复，只改变"何时切换 + 等待方式"：
       H0  立即切换（丢弃剩余 chunk）
       H1  继续执行完 A 当前 chunk 的剩余动作，再切换
       H2  保持最后电机目标等待 0.6 s 再切换
       H3  保持最后电机目标等待 1.2 s 再切换
  3. 用统一判据执行 B/C：首次满足后停止发新动作，静置 1.0 s，
     复核位置/朝向/released/稳定后判定。

用法：
    PYTHONPATH=src venv/bin/python -u scripts/eval_handoff_timing.py \
        --chain A0_C --conditions H0 H1 H2 H3 --episodes 20 \
        --out results/experiment_2_2/timing
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pybullet as p

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from eval_multi_atomic_vla import PolicyRunner  # noqa: E402
from manipulation import finger_widths, set_gripper  # noqa: E402
from vla.atomic_tasks.config2 import (  # noqa: E402
    CAMERAS, IMAGE_SIZE, STEPS_PER_ACTION, TASKS,
)
from vla.atomic_tasks.experts import axis_error_x, axis_error_y  # noqa: E402
from vla.atomic_tasks.scene2 import (  # noqa: E402
    BLOCK_HALF, PLACE, TRAY, WALL_INNER_X, block_yaw, build_scene,
)
from vla.cameras import CameraRig  # noqa: E402

CONDITIONS = ['H0', 'H1', 'H2', 'H3']
WAIT = {'H2': 0.6, 'H3': 1.2}
try:  # lerobot 的动作队列键
    from lerobot.constants import ACTION as _ACTION_KEY  # type: ignore
except Exception:  # noqa: BLE001
    _ACTION_KEY = 'action'


def obj_state(sim, block):
    pos, quat = p.getBasePositionAndOrientation(block, physicsClientId=sim.client)
    lin, ang = p.getBaseVelocity(block, physicsClientId=sim.client)
    euler = p.getEulerFromQuaternion(quat)
    return {
        'position': np.asarray(pos, dtype=float),
        'quaternion': np.asarray(quat, dtype=float),
        'yaw': block_yaw(block, sim.client),
        'tilt': abs(math.degrees(euler[0])) + abs(math.degrees(euler[1])),
        'speed': float(np.linalg.norm(lin)),
        'angular_speed': float(np.linalg.norm(ang)),
    }


def finger_contacts(sim, block):
    contacts = p.getContactPoints(bodyA=block, bodyB=sim.robot, physicsClientId=sim.client)
    return [c for c in contacts if c[4] in (9, 10) and c[9] > 0.0]


def a_success(skill, sim, block, initial_z, max_z):
    st = obj_state(sim, block)
    err = float(np.linalg.norm(st['position'][:2] - np.asarray(PLACE[:2])))
    yaw_ok = (axis_error_y(st['yaw']) <= math.radians(12) if skill == 'A0'
              else axis_error_x(st['yaw']) <= math.radians(12))
    return (err <= 0.02 and yaw_ok and st['tilt'] <= 12
            and not finger_contacts(sim, block) and (max_z - initial_z) >= 0.03
            and st['speed'] < 0.01)


def task_ok(skill, st, y0, yaw0):
    if skill == 'B':
        depth = BLOCK_HALF[0] * abs(math.cos(st['yaw'])) + BLOCK_HALF[1] * abs(math.sin(st['yaw']))
        gap = WALL_INNER_X - (st['position'][0] + depth)
        dyaw = abs(math.atan2(math.sin(st['yaw'] - yaw0), math.cos(st['yaw'] - yaw0)))
        return (0.0 <= gap <= 0.02 and abs(st['position'][1] - y0) <= 0.02
                and dyaw <= math.radians(15) and st['speed'] < 0.01)
    dx = st['position'][0] - TRAY[0]
    dy = st['position'][1] - TRAY[1]
    return (abs(dx) <= 0.02 and abs(dy) <= 0.05
            and axis_error_y(st['yaw']) <= math.radians(12)
            and st['tilt'] <= 12 and st['speed'] < 0.01)


def final_ok(skill, final, released, stable, y0, yaw0):
    if skill == 'B':
        depth = BLOCK_HALF[0] * abs(math.cos(final['yaw'])) + \
            BLOCK_HALF[1] * abs(math.sin(final['yaw']))
        gap = WALL_INNER_X - (final['position'][0] + depth)
        dyaw = abs(math.atan2(math.sin(final['yaw'] - yaw0), math.cos(final['yaw'] - yaw0)))
        return (0.0 <= gap <= 0.02 and abs(final['position'][1] - y0) <= 0.02
                and dyaw <= math.radians(15) and final['tilt'] <= 12 and stable)
    dx = final['position'][0] - TRAY[0]
    dy = final['position'][1] - TRAY[1]
    return (abs(dx) <= 0.02 and abs(dy) <= 0.05
            and axis_error_y(final['yaw']) <= math.radians(12)
            and final['tilt'] <= 12 and released and stable)


def branch_snapshot(runner, skill, seed, out: Path, args):
    """重跑 A，保存 t_a_success 的分支快照与剩余动作。"""
    sim, block, wall, tray, info = build_scene(seed=seed, jitter=0.008)
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
        last_action = None
        release_tick = None
        a_ticks = 0
        for tick in range(args.max_ticks_a):
            frames = camera.capture()
            state = np.concatenate([sim.q(), np.asarray(finger_widths(sim))])
            action = runner.act(frames, state, TASKS[skill])
            last_action = action
            target = np.clip(action[:7], np.asarray(sim.lower) + 1e-4,
                             np.asarray(sim.upper) - 1e-4)
            width = float(np.clip(action[7], 0.0, 0.04))
            sim.command(target)
            set_gripper(sim, width, 20.0, 0.05)
            for _ in range(STEPS_PER_ACTION):
                sim.step()
            a_ticks += 1
            pos, _ = p.getBasePositionAndOrientation(block, physicsClientId=sim.client)
            max_z = max(max_z, float(pos[2]))
            if release_tick is None and tick > 20 and not finger_contacts(sim, block) \
                    and width > 0.03:
                release_tick = tick
            if tick > 40 and a_success(skill, sim, block, initial_z, max_z):
                break
        else:
            return None
        queue = list(getattr(runner.policy, '_queues', {}).get(_ACTION_KEY, []))
        remaining = [np.asarray(t.detach().cpu()).reshape(-1) for t in queue]
        bullet = out / f'branch_{skill}_{seed:08d}.bullet'
        p.saveBullet(str(bullet))
        return {
            'bullet': str(bullet.resolve()),
            'seed': int(seed),
            'a_ticks': int(a_ticks),
            'release_tick': release_tick,
            'a_success_tick': int(a_ticks),
            'remaining_queue': int(len(remaining)),
            'last_action': [float(v) for v in last_action],
            'robot': {'q': [float(v) for v in sim.q()]},
            'object': {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                       for k, v in obj_state(sim, block).items()},
            'max_z': float(max_z),
        }, remaining
    finally:
        if p.isConnected(sim.client):
            p.disconnect(sim.client)


def run_condition(runner, chain, branch, remaining, cond, args):
    a_skill, b_skill = chain.split('_')
    seed = branch['seed']
    sim, block, wall, tray, info = build_scene(seed=seed, jitter=0.008)
    try:
        camera = CameraRig(sim, CAMERAS, IMAGE_SIZE, supersample=1)
        p.restoreState(fileName=branch['bullet'], physicsClientId=sim.client)
        last = np.asarray(branch['last_action'], dtype=float)
        sim.command(last[:7])
        set_gripper(sim, float(last[7]), 20.0, 0.05)
        record = {'chain': chain, 'condition': cond, 'seed': seed,
                  'branch_a_ticks': branch['a_ticks'],
                  'remaining_queue': branch['remaining_queue']}
        extra = 0
        if cond == 'H1':
            # 执行完 A 当前 chunk 的剩余动作
            for action in remaining:
                target = np.clip(np.asarray(action[:7], dtype=float),
                                 np.asarray(sim.lower) + 1e-4, np.asarray(sim.upper) - 1e-4)
                width = float(np.clip(action[7], 0.0, 0.04))
                sim.command(target)
                set_gripper(sim, width, 20.0, 0.05)
                for _ in range(STEPS_PER_ACTION):
                    sim.step()
                extra += 1
            last = remaining[-1] if len(remaining) else last
        elif cond in WAIT:
            ticks = int(round(WAIT[cond] / (STEPS_PER_ACTION * sim.dt)))
            for _ in range(ticks):
                sim.command(last[:7])
                set_gripper(sim, float(last[7]), 20.0, 0.05)
                for _ in range(STEPS_PER_ACTION):
                    sim.step()
                extra += 1
        record['extra_ticks'] = extra
        st = obj_state(sim, block)
        record['switch_object'] = {'position': st['position'].tolist(), 'yaw': float(st['yaw']),
                                   'speed': st['speed']}
        record['switch_robot_q'] = [float(v) for v in sim.q()]
        y0 = float(st['position'][1])
        yaw0 = st['yaw']

        runner.reset()
        if not args.no_reseed:
            np.random.seed(seed)
            runner.torch.manual_seed(seed)
            runner.torch.cuda.manual_seed_all(seed)
        started = time.perf_counter()
        success = False
        first_success = None
        tick = 0
        last_target = np.asarray(sim.q(), dtype=float)
        last_width = 0.04
        for tick in range(args.max_ticks_b):
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
            st = obj_state(sim, block)
            released_now = not finger_contacts(sim, block)
            if tick > 15 and task_ok(b_skill, st, y0, yaw0) \
                    and (b_skill == 'B' or released_now):
                success = True
                first_success = tick
                break
        # 统一判定：停止发新动作（保持最后一条命令），静置 1.0 s 后复核
        for _ in range(int(round(1.0 / sim.dt))):
            sim.command(last_target)
            set_gripper(sim, last_width, 20.0, 0.05)
            sim.step()
        stable = True
        for _ in range(5):
            s2 = obj_state(sim, block)
            if s2['speed'] >= 0.01:
                stable = False
            for _ in range(int(round(0.04 / sim.dt))):
                sim.step()
        final = obj_state(sim, block)
        released = not finger_contacts(sim, block)
        good = bool(success and final_ok(b_skill, final, released, stable, y0, yaw0))
        record.update({
            'success': good, 'first_success_tick': first_success,
            'ticks': int(tick + 1), 'wall_seconds': round(time.perf_counter() - started, 2),
            'released': bool(released), 'stable': bool(stable),
            'final_object': {'position': final['position'].tolist(),
                             'yaw': float(final['yaw']), 'tilt_deg': float(final['tilt'])},
            'failure': None if good else ('timeout' if first_success is None else 'final_check'),
        })
        return record
    finally:
        if p.isConnected(sim.client):
            p.disconnect(sim.client)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', default='outputs/smolvla_multi/checkpoints/020000/pretrained_model')
    ap.add_argument('--snapshots', type=Path, default=ROOT / 'results/experiment_2_1/snapshots')
    ap.add_argument('--chain', default='A0_C')
    ap.add_argument('--conditions', nargs='+', default=CONDITIONS)
    ap.add_argument('--episodes', type=int, default=20)
    ap.add_argument('--max-ticks-a', type=int, default=200)
    ap.add_argument('--max-ticks-b', type=int, default=260)
    ap.add_argument('--warmup-steps', type=int, default=30)
    ap.add_argument('--from-snapshot', action='store_true',
                    help='直接用 2.1 快照作为分支点（不再重跑 A），用于核对 C 循环与 2.1 是否一致')
    ap.add_argument('--no-reseed', action='store_true',
                    help='切换后不重置 RNG（复现实验 2 的连续 RNG 流）')
    ap.add_argument('--out', type=Path, default=ROOT / 'results/experiment_2_2/timing')
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    a_skill, b_skill = args.chain.split('_')
    snaps = []
    for path in sorted(args.snapshots.glob(f'{a_skill}_*.json')):
        if 'restore' in path.name or 'collection' in path.name:
            continue
        snaps.append(json.loads(path.read_text(encoding='utf-8')))
    snaps = snaps[:args.episodes]

    runner = PolicyRunner(args.policy)
    rows = []
    for i, snap in enumerate(snaps):
        if args.from_snapshot:
            branch = {
                'bullet': str((args.snapshots / snap['bullet_file']).resolve()),
                'seed': snap['seed'],
                'a_ticks': -1, 'release_tick': None, 'a_success_tick': -1,
                'remaining_queue': 0, 'last_action': snap['last_action'],
                'robot': {'q': snap['robot']['q']}, 'object': snap['object'],
                'max_z': snap['object']['position'][2],
            }
            remaining = []
        else:
            branch, remaining = branch_snapshot(runner, a_skill, snap['seed'], args.out, args)
        if branch is None:
            print(f'[{args.chain} {i + 1}/{len(snaps)}] seed={snap["seed"]} A branch failed')
            continue
        for cond in args.conditions:
            rec = run_condition(runner, args.chain, branch, remaining, cond, args)
            rows.append(rec)
            print(f'[{args.chain} {i + 1}/{len(snaps)}] {cond} seed={snap["seed"]} '
                  f'{"OK" if rec["success"] else "FAIL"} extra={rec["extra_ticks"]} '
                  f'switch_speed={rec["switch_object"]["speed"]:.4f} '
                  f'ticks={rec["ticks"]}', flush=True)

    summary = {}
    for cond in args.conditions:
        group = [r for r in rows if r['condition'] == cond]
        ok = sum(1 for r in group if r['success'])
        summary[cond] = {'n': len(group), 'success': ok,
                         'rate': round(ok / max(len(group), 1), 4),
                         'extra_ticks_mean': float(np.mean([r['extra_ticks'] for r in group]))}
    (args.out / f'timing_{args.chain}.json').write_text(
        json.dumps({'chain': args.chain, 'summary': summary, 'episodes': rows},
                   ensure_ascii=False, indent=2), encoding='utf-8')
    fields = ['chain', 'condition', 'seed', 'success', 'failure', 'first_success_tick',
              'ticks', 'extra_ticks', 'branch_a_ticks', 'remaining_queue', 'released',
              'stable', 'wall_seconds']
    with (args.out / f'runs_{args.chain}.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({'chain': args.chain, 'summary': summary}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
