"""实验 2.1 阶段 2：用冻结的 SmolVLA 跑 A0/A90，保存真实交接状态快照。

只保存 A 真实成功（抬升 >=3 cm、自主释放、稳定）后的状态；
不修改物体、不调用专家。

用法：
    PYTHONPATH=src venv/bin/python -u scripts/collect_handoff_snapshots.py \
        --skill A0 --episodes 20 --seed-base 40000000 \
        --out results/experiment_2_1/snapshots
"""
from __future__ import annotations

import argparse
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
from vla.atomic_tasks.scene2 import PLACE, block_yaw, build_scene  # noqa: E402
from vla.cameras import CameraRig  # noqa: E402


def finger_contact_count(sim, block):
    contacts = p.getContactPoints(bodyA=block, bodyB=sim.robot, physicsClientId=sim.client)
    return sum(1 for c in contacts if c[4] in (9, 10) and c[9] > 0.0)


def table_contact(sim, block):
    contacts = p.getContactPoints(bodyA=block, bodyB=sim.plane, physicsClientId=sim.client)
    return len(contacts)


def object_info(sim, block):
    pos, quat = p.getBasePositionAndOrientation(block, physicsClientId=sim.client)
    lin, ang = p.getBaseVelocity(block, physicsClientId=sim.client)
    euler = p.getEulerFromQuaternion(quat)
    tilt = abs(math.degrees(euler[0])) + abs(math.degrees(euler[1]))
    return {
        'position': [float(v) for v in pos],
        'quaternion': [float(v) for v in quat],
        'linear_velocity': [float(v) for v in lin],
        'angular_velocity': [float(v) for v in ang],
        'yaw_rad': block_yaw(block, sim.client),
        'tilt_deg': float(tilt),
    }


def robot_info(sim):
    states = p.getJointStates(sim.robot, list(range(7)), physicsClientId=sim.client)
    q = [float(s[0]) for s in states]
    qd = [float(s[1]) for s in states]
    ee_pos, ee_quat = sim.pose()
    fingers = p.getJointStates(sim.robot, [9, 10], physicsClientId=sim.client)
    return {
        'q': q, 'qd': qd,
        'ee_position': [float(v) for v in ee_pos],
        'ee_quaternion': [float(v) for v in ee_quat],
        'finger_positions': [float(f[0]) for f in fingers],
        'finger_velocities': [float(f[1]) for f in fingers],
        'limit_margin_lower': [float(v) for v in (np.asarray(q) - sim.lower)],
        'limit_margin_upper': [float(v) for v in (sim.upper - np.asarray(q))],
    }


def a_success(skill, sim, block, initial_z, max_z):
    st = object_info(sim, block)
    pos = np.asarray(st['position'])
    err = float(np.linalg.norm(pos[:2] - np.asarray(PLACE[:2])))
    yaw_ok = (axis_error_y(st['yaw_rad']) <= math.radians(12) if skill == 'A0'
              else axis_error_x(st['yaw_rad']) <= math.radians(12))
    lift = float(max_z - initial_z)
    released = finger_contact_count(sim, block) == 0
    speed = float(np.linalg.norm(st['linear_velocity']))
    ok = (err <= 0.02 and yaw_ok and st['tilt_deg'] <= 12 and released and lift >= 0.03
          and speed < 0.01)
    return ok, {'xy_error_m': err, 'yaw_rad': st['yaw_rad'], 'lift_m': lift,
                'released': released, 'speed': speed, 'tilt_deg': st['tilt_deg']}


def run_episode(runner, skill, seed, out: Path, args):
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
        started = time.perf_counter()
        tick = 0
        for tick in range(args.max_ticks):
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
            pos, _ = p.getBasePositionAndOrientation(block, physicsClientId=sim.client)
            max_z = max(max_z, float(pos[2]))
            if (release_tick is None and tick > 20 and finger_contact_count(sim, block) == 0
                    and width > 0.03):
                release_tick = tick
            if tick > 40:
                ok, detail = a_success(skill, sim, block, initial_z, max_z)
                if ok and release_tick is not None:
                    break
        # 判定后静置 1.0 s（不发送新指令）
        for _ in range(int(round(args.settle_seconds / sim.dt))):
            sim.step()
        ok, detail = a_success(skill, sim, block, initial_z, max_z)
        # 连续 5 次速度检查
        stable = True
        for _ in range(5):
            st = object_info(sim, block)
            if float(np.linalg.norm(st['linear_velocity'])) >= 0.01:
                stable = False
            for _ in range(int(round(0.04 / sim.dt))):
                sim.step()
        ok = bool(ok and stable)
        record = {
            'skill': skill, 'seed': int(seed), 'success': ok,
            'ticks': int(tick + 1), 'max_z': float(max_z), 'initial_z': initial_z,
            'lift_m': float(max_z - initial_z), 'release_tick': release_tick,
            'wall_seconds': round(time.perf_counter() - started, 2),
            'detail': detail,
        }
        if not ok:
            return record, None

        # 保存前重新渲染：必须与快照状态同一时刻，不能复用循环里的旧帧。
        shot = {name: np.asarray(camera.capture()[name], dtype=np.uint8).copy()
                for name in CAMERAS}
        state_vec = np.concatenate([sim.q(), np.asarray(finger_widths(sim))])
        bullet = out / f'{skill}_{seed:08d}.bullet'
        p.saveBullet(str(bullet))
        snap = {
            **record,
            'robot': robot_info(sim),
            'object': object_info(sim, block),
            'finger_contacts': finger_contact_count(sim, block),
            'table_contacts': table_contact(sim, block),
            'policy_state': [float(v) for v in state_vec],
            'last_action': [float(v) for v in last_action],
            'task': TASKS[skill],
            'next_skills': ['B', 'C'],
            'scene': {'jitter': 0.008},
            'bullet_file': bullet.name,
            'top_png': f'{skill}_{seed:08d}_top.png',
            'wrist_png': f'{skill}_{seed:08d}_wrist.png',
        }
        from PIL import Image
        Image.fromarray(shot['top']).save(out / snap['top_png'])
        Image.fromarray(shot['wrist']).save(out / snap['wrist_png'])
        (out / f'{skill}_{seed:08d}.json').write_text(
            json.dumps(snap, ensure_ascii=False, indent=2), encoding='utf-8')
        return record, snap
    finally:
        if p.isConnected(sim.client):
            p.disconnect(sim.client)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', default='outputs/smolvla_multi/checkpoints/020000/pretrained_model')
    ap.add_argument('--skills', nargs='+', default=['A0', 'A90'])
    ap.add_argument('--episodes', type=int, default=20)
    ap.add_argument('--seed-base', type=int, default=40000000)
    ap.add_argument('--max-ticks', type=int, default=200)
    ap.add_argument('--warmup-steps', type=int, default=30)
    ap.add_argument('--settle-seconds', type=float, default=1.0)
    ap.add_argument('--out', default='results/experiment_2_1/snapshots')
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runner = PolicyRunner(args.policy)
    log = []
    for skill in args.skills:
        for i in range(args.episodes):
            seed = args.seed_base + i
            record, _ = run_episode(runner, skill, seed, out, args)
            log.append(record)
            print(f'[{skill} {i + 1}/{args.episodes}] seed={seed} '
                  f'{"SAVED" if record["success"] else "FAIL"} '
                  f'ticks={record["ticks"]} lift={record["lift_m"]:.3f}', flush=True)
    (out / 'collection_log.json').write_text(json.dumps(log, ensure_ascii=False, indent=2),
                                             encoding='utf-8')
    saved = sum(1 for r in log if r['success'])
    print(f'saved {saved}/{len(log)} snapshots -> {out}')


if __name__ == '__main__':
    main()
