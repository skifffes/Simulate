"""实验 2.1 阶段 3：交接状态控制变量诊断（T0/T1/T2/T3）。

在每个真实 A 终态快照上构造四种条件，用冻结 SmolVLA 执行后继技能 B/C：

    T0 真实交接：机器人 + 物体都保持 A 终态
    T1 只恢复机器人：机器人回 HOME，物体保持 A 终态
    T2 只恢复物体：机器人保持 A 终态，物体回标准起点 (PLACE, yaw 0)
    T3 共同标准状态：机器人和物体都回标准

所有构造只发生在离线诊断开始之前；执行阶段完全由 VLA 控制。
构造后会做物理有效性检查（关节限位、穿插、物体高度、静止），无效条件记录但不运行。

用法：
    PYTHONPATH=src venv/bin/python -u scripts/diagnose_handoff_states.py \
        --chains A90_C --conditions T0 T1 T2 T3 --out results/experiment_2_1/diagnosis
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
from simulation import HOME  # noqa: E402
from vla.cameras import CameraRig  # noqa: E402

CONDITIONS = ['T0', 'T1', 'T2', 'T3']


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


def construct(sim, block, snap, cond, settle_steps):
    if cond in ('T1', 'T3'):
        for j, value in enumerate(HOME):
            p.resetJointState(sim.robot, j, float(value), 0.0, physicsClientId=sim.client)
        for j in (9, 10):
            p.resetJointState(sim.robot, j, 0.035, 0.0, physicsClientId=sim.client)
        sim.command(np.asarray(HOME, dtype=float))
        set_gripper(sim, 0.04, 20.0, 0.06)
    else:
        sim.command(np.asarray(snap['last_action'][:7], dtype=float))
        set_gripper(sim, float(snap['last_action'][7]), 20.0, 0.05)
    if cond in ('T2', 'T3'):
        p.resetBasePositionAndOrientation(
            block, list(PLACE), p.getQuaternionFromEuler([0.0, 0.0, 0.0]),
            physicsClientId=sim.client)
        p.resetBaseVelocity(block, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                            physicsClientId=sim.client)
    for _ in range(int(settle_steps)):
        sim.step()
    # 不推进物理，只刷新窄相碰撞，保证 T0 不被额外步进扰动。
    p.performCollisionDetection(physicsClientId=sim.client)


def validity(sim, block):
    st = obj_state(sim, block)
    contacts = p.getContactPoints(bodyA=block, bodyB=sim.robot, physicsClientId=sim.client)
    penetration = min([float(c[8]) for c in contacts], default=0.0)
    q = sim.q()
    joints_ok = bool(np.all(q >= sim.lower - 1e-3) and np.all(q <= sim.upper + 1e-3))
    table = p.getContactPoints(bodyA=block, bodyB=sim.plane, physicsClientId=sim.client)
    info = {
        'object_height_m': float(st['position'][2]),
        'object_speed': st['speed'],
        'min_robot_object_distance': penetration,
        'table_contacts': len(table),
        'joints_ok': joints_ok,
    }
    ok = (0.010 <= st['position'][2] <= 0.035 and penetration > -0.002
          and st['speed'] < 0.05 and joints_ok)
    return ok, info


def b_done(st, y0, yaw0):
    depth = BLOCK_HALF[0] * abs(math.cos(st['yaw'])) + BLOCK_HALF[1] * abs(math.sin(st['yaw']))
    gap = WALL_INNER_X - (st['position'][0] + depth)
    dyaw = abs(math.atan2(math.sin(st['yaw'] - yaw0), math.cos(st['yaw'] - yaw0)))
    return (0.0 <= gap <= 0.02 and abs(st['position'][1] - y0) <= 0.02
            and dyaw <= math.radians(15) and st['speed'] < 0.01)


def c_done(st):
    dx = st['position'][0] - TRAY[0]
    dy = st['position'][1] - TRAY[1]
    return (abs(dx) <= 0.02 and abs(dy) <= 0.05
            and axis_error_y(st['yaw']) <= math.radians(12)
            and st['tilt'] <= 12 and st['speed'] < 0.01)


def run_condition(runner, chain, snap, cond, args):
    a_skill, b_skill = chain.split('_')
    sim, block, wall, tray, info = build_scene(seed=snap['seed'], jitter=0.008)
    try:
        camera = CameraRig(sim, CAMERAS, IMAGE_SIZE, supersample=1)
        p.restoreState(fileName=str(args.snapshots / snap['bullet_file']),
                       physicsClientId=sim.client)
        construct(sim, block, snap, cond, args.settle_steps)
        st = obj_state(sim, block)
        y0 = float(st['position'][1])
        yaw0 = st['yaw']
        start_z = float(st['position'][2])
        ok_valid, valid_info = validity(sim, block)
        record = {
            'chain': chain, 'condition': cond, 'seed': snap['seed'],
            'valid': bool(ok_valid), 'validity': valid_info,
            'start_object': {'position': st['position'].tolist(), 'yaw': yaw0},
        }
        if not ok_valid:
            record.update({'success': False, 'failure': 'invalid_init', 'ticks': 0})
            return record, None

        runner.reset()
        np.random.seed(snap['seed'])
        runner.torch.manual_seed(snap['seed'])
        runner.torch.cuda.manual_seed_all(snap['seed'])
        events = {}
        max_z = float(st['position'][2])
        started = time.perf_counter()
        tick = 0
        success = False
        for tick in range(args.max_ticks):
            frames = camera.capture()
            state = np.concatenate([sim.q(), np.asarray(finger_widths(sim))])
            action = runner.act(frames, state, TASKS[b_skill])
            target = np.clip(action[:7], np.asarray(sim.lower) + 1e-4,
                             np.asarray(sim.upper) - 1e-4)
            width = float(np.clip(action[7], 0.0, 0.04))
            sim.command(target)
            set_gripper(sim, width, 20.0, 0.05)
            for _ in range(STEPS_PER_ACTION):
                sim.step()
            st = obj_state(sim, block)
            max_z = max(max_z, float(st['position'][2]))
            contacts = finger_contacts(sim, block)
            if 'first_contact' not in events and contacts:
                events['first_contact'] = tick
            if 'close' not in events and width <= 0.025 and tick > 2:
                events['close'] = tick
            if 'lift' not in events and st['position'][2] > 0.03:
                events['lift'] = tick
            if 'release' not in events and 'close' in events and width >= 0.03 \
                    and not contacts:
                events['release'] = tick
            if tick > 15:
                if b_skill == 'B':
                    success = b_done(st, y0, yaw0)
                else:
                    success = c_done(st)
                # 不提前截断：跑满整个时间窗，最终判定在 1 s 静置后统一做，
                # 否则会在“刚进托盘但还没松爪”时截断，把释放误判成失败。
                if success and 'first_success' not in events:
                    events['first_success'] = tick
        for _ in range(int(round(args.settle_seconds / sim.dt))):
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
        if b_skill == 'B':
            depth = BLOCK_HALF[0] * abs(math.cos(final['yaw'])) + \
                BLOCK_HALF[1] * abs(math.sin(final['yaw']))
            gap = WALL_INNER_X - (final['position'][0] + depth)
            dyaw = abs(math.atan2(math.sin(final['yaw'] - yaw0),
                                  math.cos(final['yaw'] - yaw0)))
            good = (0.0 <= gap <= 0.02 and abs(final['position'][1] - y0) <= 0.02
                    and dyaw <= math.radians(15) and final['tilt'] <= 12 and stable)
            detail = {'gap_m': gap, 'lateral_dy_m': float(final['position'][1] - y0),
                      'yaw_change_rad': dyaw}
        else:
            dx = float(final['position'][0] - TRAY[0])
            dy = float(final['position'][1] - TRAY[1])
            good = (abs(dx) <= 0.02 and abs(dy) <= 0.05
                    and axis_error_y(final['yaw']) <= math.radians(12)
                    and final['tilt'] <= 12 and released and stable)
            detail = {'tray_dx_m': dx, 'tray_dy_m': dy}
        record.update({
            'success': bool(good), 'ticks': int(tick + 1),
            'wall_seconds': round(time.perf_counter() - started, 2),
            'lift_m': float(max_z - start_z),
            'max_z': float(max_z), 'events': events,
            'final_object': {'position': final['position'].tolist(),
                             'yaw': float(final['yaw']),
                             'tilt_deg': float(final['tilt']),
                             'speed': final['speed']},
            'released': bool(released), 'stable': bool(stable),
            'detail': detail,
        })
        record['failure'] = None if good else ('push_check' if b_skill == 'B' else 'tray_check')
        return record, None
    finally:
        if p.isConnected(sim.client):
            p.disconnect(sim.client)


def load_snapshots(root: Path, a_skill: str):
    snaps = []
    for path in sorted(root.glob(f'{a_skill}_*.json')):
        if 'restore' in path.name or 'collection' in path.name:
            continue
        snaps.append(json.loads(path.read_text(encoding='utf-8')))
    return snaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', default='outputs/smolvla_multi/checkpoints/020000/pretrained_model')
    ap.add_argument('--snapshots', type=Path, default=ROOT / 'results/experiment_2_1/snapshots')
    ap.add_argument('--chains', nargs='+', default=['A90_C'])
    ap.add_argument('--conditions', nargs='+', default=CONDITIONS)
    ap.add_argument('--episodes', type=int, default=20)
    ap.add_argument('--max-ticks', type=int, default=260)
    ap.add_argument('--settle-steps', type=int, default=0,
                    help='构造后的额外 settle 步数；T0 必须为 0 才能复现真实交接，'
                         '否则会把机器臂拉回旧目标、扰动物体')
    ap.add_argument('--settle-seconds', type=float, default=1.0)
    ap.add_argument('--out', type=Path, default=ROOT / 'results/experiment_2_1/diagnosis')
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    runner = PolicyRunner(args.policy)
    all_rows = []
    for chain in args.chains:
        a_skill, b_skill = chain.split('_')
        snaps = load_snapshots(args.snapshots, a_skill)[:args.episodes]
        rows = []
        for i, snap in enumerate(snaps):
            for cond in args.conditions:
                rec, _ = run_condition(runner, chain, snap, cond, args)
                rec['snapshot'] = snap.get('seed')
                rows.append(rec)
                all_rows.append(rec)
                print(f'[{chain} {i + 1}/{len(snaps)}] {cond} seed={snap["seed"]} '
                      f'{"OK" if rec["success"] else "FAIL"} '
                      f'valid={rec["valid"]} ticks={rec["ticks"]} '
                      f'events={rec.get("events")}', flush=True)
        summary = {}
        for cond in args.conditions:
            group = [r for r in rows if r['condition'] == cond]
            valid = [r for r in group if r['valid']]
            summary[cond] = {
                'n': len(group), 'valid': len(valid),
                'success': sum(1 for r in valid if r['success']),
                'rate': round(sum(1 for r in valid if r['success']) / max(len(valid), 1), 4),
            }
        (args.out / f'diagnosis_{chain}.json').write_text(
            json.dumps({'chain': chain, 'summary': summary, 'episodes': rows},
                       ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'chain': chain, 'summary': summary}, ensure_ascii=False), flush=True)

    fields = ['chain', 'condition', 'seed', 'valid', 'success', 'failure', 'ticks',
              'lift_m', 'max_z', 'released', 'stable', 'wall_seconds']
    with (args.out / 'runs.csv').open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_rows)
    print('done', len(all_rows))


if __name__ == '__main__':
    main()
