"""实验 2.2 阶段 3：Physical Retreat to HOME 基线。

从 A 的真实终态（2.1 快照）出发，用**连续关节运动**把 Panda 送回 HOME，
不调用 resetJointState、不瞬移；只改机器人构型，不碰物体。

流程：
  1. 恢复快照，夹爪保持张开；
  2. 规划并执行退让路径（优先直接关节空间；若碰撞则"先抬升再回 HOME"）；
  3. 检查：到达 HOME、无碰撞、物体位移 < 5 mm、A 的任务仍成功；
  4. 用冻结的 B/C VLA 执行后继任务（统一判据 v2.2）。

用法：
    PYTHONPATH=src venv/bin/python -u scripts/eval_physical_retreat.py \
        --chain A90_C --episodes 20 --out results/experiment_2_2/retreat
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

from eval_handoff_timing import final_ok, finger_contacts, obj_state, task_ok  # noqa: E402
from eval_multi_atomic_vla import PolicyRunner  # noqa: E402
from manipulation import finger_widths, set_gripper  # noqa: E402
from robot_controller import RobotController  # noqa: E402
from simulation import HOME  # noqa: E402
from vla.atomic_tasks.config2 import CAMERAS, IMAGE_SIZE, STEPS_PER_ACTION, TASKS  # noqa: E402
from vla.atomic_tasks.experts import axis_error_x, axis_error_y  # noqa: E402
from vla.atomic_tasks.scene2 import PLACE, block_yaw, build_scene  # noqa: E402
from vla.cameras import CameraRig  # noqa: E402


def a_still_ok(skill, so, initial, obj0):
    err = float(np.linalg.norm(np.asarray(so['position'][:2]) - np.asarray(PLACE[:2])))
    yaw_ok = (axis_error_y(so['yaw']) <= math.radians(12) if skill == 'A0'
              else axis_error_x(so['yaw']) <= math.radians(12))
    moved = float(np.linalg.norm(np.asarray(obj0['position']) - np.asarray(so['position'])))
    return (err <= 0.02 and yaw_ok and so['tilt'] <= 12 and moved <= 0.005
            and so['speed'] < 0.01), err, moved


def execute_joint_path(sim, block, waypoints, args, stats):
    q = sim.q().copy()
    for wp in waypoints:
        delta = np.asarray(wp, dtype=float) - q
        n = max(1, int(np.ceil(float(np.linalg.norm(delta)) / args.joint_step)))
        for k in range(1, n + 1):
            sim.command(q + delta * (k / n))
            sim.step()
            stats['ticks'] += 1
            if sim.collisions():
                stats['collision'] = True
                return False, 'collision'
        q = np.asarray(wp, dtype=float)
    return True, 'ok'


def retreat(sim, block, args, stats):
    """返回 (ok, mode, reason)。优先直接关节路径，失败则抬升后再回。"""
    q0 = sim.q().copy()
    home = np.asarray(HOME, dtype=float)
    # 1) 直接关节空间
    ok, reason = execute_joint_path(sim, block, [home], args, stats)
    if ok:
        return True, 'direct', 'ok'
    # 2) 回滚到快照，抬升 8 cm 后再回 HOME
    p.restoreState(fileName=args._bullet, physicsClientId=sim.client)
    sim.command(q0)
    set_gripper(sim, 0.04, 20.0, 0.06)
    for _ in range(5):
        sim.step()
    q0 = sim.q().copy()
    ctl = RobotController(sim)
    ee_pos, ee_quat = sim.pose()
    lift_target = np.asarray(ee_pos, dtype=float) + np.array([0.0, 0.0, 0.08])
    q_lift, ok_ik, _ = ctl.solve(lift_target, ee_quat)
    if not ok_ik:
        return False, 'staged', 'ik_lift'
    stats['ticks'] = 0
    stats['collision'] = False
    ok, reason = execute_joint_path(sim, block, [q_lift, home], args, stats)
    return ok, 'staged', reason


def run_vla(runner, chain, sim, block, args):
    b_skill = chain.split('_')[1]
    camera = CameraRig(sim, CAMERAS, IMAGE_SIZE, supersample=1)
    st = obj_state(sim, block)
    y0, yaw0 = float(st['position'][1]), st['yaw']
    runner.reset()
    seed = args.seed
    np.random.seed(seed)
    runner.torch.manual_seed(seed)
    runner.torch.cuda.manual_seed_all(seed)
    started = time.perf_counter()
    success = False
    first_success = None
    last_target = np.asarray(sim.q(), dtype=float)
    last_width = 0.04
    tick = 0
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
        st2 = obj_state(sim, block)
        released_now = not finger_contacts(sim, block)
        if tick > 15 and task_ok(b_skill, st2, y0, yaw0) and (b_skill == 'B' or released_now):
            success = True
            first_success = tick
            break
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
    return {
        'vla_success': good, 'first_success_tick': first_success,
        'vla_ticks': int(tick + 1), 'vla_wall_seconds': round(time.perf_counter() - started, 2),
        'released': bool(released), 'stable': bool(stable),
        'final_object': {'position': final['position'].tolist(),
                         'yaw': float(final['yaw']), 'tilt_deg': float(final['tilt'])},
        'vla_failure': None if good else ('timeout' if first_success is None else 'final_check'),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', default='outputs/smolvla_multi/checkpoints/020000/pretrained_model')
    ap.add_argument('--snapshots', type=Path, default=ROOT / 'results/experiment_2_1/snapshots')
    ap.add_argument('--chain', default='A90_C')
    ap.add_argument('--episodes', type=int, default=20)
    ap.add_argument('--joint-step', type=float, default=0.002)
    ap.add_argument('--max-ticks-b', type=int, default=260)
    ap.add_argument('--out', type=Path, default=ROOT / 'results/experiment_2_2/retreat')
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
        args.seed = snap['seed']
        args._bullet = str((args.snapshots / snap['bullet_file']).resolve())
        sim, block, wall, tray, info = build_scene(seed=snap['seed'], jitter=0.008)
        try:
            p.restoreState(fileName=args._bullet, physicsClientId=sim.client)
            sim.command(np.asarray(snap['last_action'][:7], dtype=float))
            set_gripper(sim, 0.04, 20.0, 0.06)
            for _ in range(5):
                sim.step()
            sim.allowed_contacts = set()  # 退让期间任何机器人接触都算碰撞
            obj_before = obj_state(sim, block)
            stats = {'ticks': 0, 'collision': False}
            t0 = time.perf_counter()
            ok, mode, reason = retreat(sim, block, args, stats)
            retreat_wall = round(time.perf_counter() - t0, 2)
            arrived = bool(np.max(np.abs(sim.q() - np.asarray(HOME))) < 0.06)
            so = obj_state(sim, block)
            a_ok, a_err, moved = a_still_ok(a_skill, so, snap['object'], obj_before)
            record = {
                'chain': args.chain, 'seed': snap['seed'],
                'retreat_ok': bool(ok and arrived), 'retreat_mode': mode,
                'retreat_reason': reason, 'retreat_ticks': stats['ticks'],
                'retreat_collision': stats['collision'],
                'joint_error_home': float(np.max(np.abs(sim.q() - np.asarray(HOME)))),
                'object_moved_m': moved, 'object_yaw_change': float(
                    abs(so['yaw'] - obj_before['yaw'])),
                'a_still_ok': bool(a_ok), 'a_xy_error_m': a_err,
                'retreat_wall_seconds': retreat_wall,
            }
            if ok and arrived and a_ok:
                record.update(run_vla(runner, args.chain, sim, block, args))
            else:
                record.update({'vla_success': False, 'vla_failure': 'retreat_failed'})
            rows.append(record)
            print(f'[{args.chain} {i + 1}/{len(snaps)}] seed={snap["seed"]} '
                  f'retreat={record["retreat_ok"]}({mode}/{reason}) '
                  f'A_ok={record["a_still_ok"]} vla={record.get("vla_success")} '
                  f'moved={moved * 1000:.1f}mm ticks={stats["ticks"]}', flush=True)
        finally:
            if p.isConnected(sim.client):
                p.disconnect(sim.client)

    n = len(rows)
    summary = {
        'chain': args.chain, 'episodes': n,
        'retreat_ok': sum(1 for r in rows if r['retreat_ok']),
        'a_still_ok': sum(1 for r in rows if r['a_still_ok']),
        'chain_success': sum(1 for r in rows if r.get('vla_success')),
        'retreat_rate': round(sum(1 for r in rows if r['retreat_ok']) / max(n, 1), 4),
        'chain_rate': round(sum(1 for r in rows if r.get('vla_success')) / max(n, 1), 4),
        'mode_counts': {m: sum(1 for r in rows if r['retreat_mode'] == m)
                        for m in ('direct', 'staged')},
        'collisions': sum(1 for r in rows if r['retreat_collision']),
        'mean_object_moved_m': float(np.mean([r['object_moved_m'] for r in rows])),
        'mean_retreat_seconds': float(np.mean([r['retreat_wall_seconds'] for r in rows])),
    }
    (args.out / f'retreat_{args.chain}.json').write_text(
        json.dumps({'summary': summary, 'episodes': rows}, ensure_ascii=False, indent=2),
        encoding='utf-8')
    fields = ['chain', 'seed', 'retreat_ok', 'retreat_mode', 'retreat_reason',
              'retreat_ticks', 'retreat_collision', 'joint_error_home', 'object_moved_m',
              'a_still_ok', 'vla_success', 'vla_failure', 'first_success_tick', 'vla_ticks',
              'retreat_wall_seconds', 'vla_wall_seconds']
    with (args.out / f'runs_{args.chain}.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
