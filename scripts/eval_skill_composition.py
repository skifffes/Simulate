"""实验 2 阶段 5：VLA→VLA 未见技能组合（A0/A90 → B/C）。

机制：
- Oracle Scheduler 用真实仿真状态判断当前技能是否完成，完成后把语言指令切到下一条；
- 切换时 `policy.reset()` 清空尚未执行的动作块；
- A 失败则整链失败；A 与后继在**同一个 episode** 里连续执行，不重置物体/机器人/夹爪。

用法：
    python scripts/eval_skill_composition.py \
        --policy outputs/smolvla_multi/checkpoints/020000/pretrained_model \
        --chains A0_B A90_B A0_C A90_C --episodes 20 \
        --out results/experiment_2/eval_composition
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from vla.atomic_tasks.config2 import CAMERAS, IMAGE_SIZE, STEPS_PER_ACTION, TASKS  # noqa: E402
from vla.atomic_tasks.experts import axis_error_x, axis_error_y  # noqa: E402
from vla.atomic_tasks.scene2 import (  # noqa: E402
    BLOCK_HALF, PLACE, TRAY, WALL_INNER_X, block_yaw, build_scene,
)
from vla.cameras import CameraRig  # noqa: E402

ACTION_DIM = 8


class PolicyRunner:
    def __init__(self, path: str, device: str = 'cuda', n_action_steps=None):
        import torch
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        self.torch = torch
        self.device = torch.device(device)
        self.policy = SmolVLAPolicy.from_pretrained(str(path))
        if n_action_steps:
            self.policy.config.n_action_steps = int(n_action_steps)
        self.policy.to(self.device)
        self.policy.eval()

    def reset(self):
        self.policy.reset()

    def act(self, frames, state, task):
        torch = self.torch
        batch = {}
        for name in CAMERAS:
            img = torch.from_numpy(np.ascontiguousarray(frames[name])).to(self.device)
            batch[f'observation.images.{name}'] = img.float().div_(255.0).permute(2, 0, 1).unsqueeze(0)
        batch['observation.state'] = (
            torch.from_numpy(np.asarray(state, dtype=np.float32)).to(self.device).unsqueeze(0))
        batch['task'] = [task]
        with torch.no_grad():
            action = self.policy.select_action(batch)
        action = action.squeeze(0).float().cpu().numpy().reshape(-1)
        if action.shape[0] < ACTION_DIM:
            action = np.concatenate([action, np.zeros(ACTION_DIM - action.shape[0])])
        return action[:ACTION_DIM]


def fingers_contact(sim, block):
    import pybullet as p
    contacts = p.getContactPoints(bodyA=block, bodyB=sim.robot, physicsClientId=sim.client)
    return sum(1 for c in contacts if c[4] in (9, 10) and c[9] > 0.0)


def block_state(sim, block):
    import pybullet as p
    pos, quat = p.getBasePositionAndOrientation(block, physicsClientId=sim.client)
    vel = p.getBaseVelocity(block, physicsClientId=sim.client)[0]
    euler = p.getEulerFromQuaternion(quat)
    return {
        'position': np.asarray(pos, dtype=float),
        'yaw': block_yaw(block, sim.client),
        'speed': float(np.linalg.norm(vel)),
        'tilt': float(abs(math.degrees(euler[0])) + abs(math.degrees(euler[1]))),
        'contacts': fingers_contact(sim, block),
    }


def a_done(skill, st):
    if st['speed'] > 0.01 or st['contacts'] > 0:
        return False
    err = float(np.linalg.norm(st['position'][:2] - np.asarray(PLACE[:2])))
    yaw_ok = (axis_error_y(st['yaw']) <= math.radians(12) if skill == 'A0'
              else axis_error_x(st['yaw']) <= math.radians(12))
    return err <= 0.02 and yaw_ok and st['tilt'] <= 12


def b_done(st, yaw0, y0):
    depth = BLOCK_HALF[0] * abs(math.cos(st['yaw'])) + BLOCK_HALF[1] * abs(math.sin(st['yaw']))
    gap = WALL_INNER_X - (st['position'][0] + depth)
    dyaw = abs(math.atan2(math.sin(st['yaw'] - yaw0), math.cos(st['yaw'] - yaw0)))
    return (0.0 <= gap <= 0.02 and abs(st['position'][1] - y0) <= 0.02
            and dyaw <= math.radians(15) and st['speed'] < 0.01)


def c_done(st):
    dx = st['position'][0] - TRAY[0]
    dy = st['position'][1] - TRAY[1]
    return (abs(dx) <= 0.02 and abs(dy) <= 0.05 and axis_error_y(st['yaw']) <= math.radians(12)
            and st['tilt'] <= 12 and st['speed'] < 0.01 and st['contacts'] == 0)


def run_chain(runner, chain: str, seed: int, args):
    import pybullet as p
    from manipulation import finger_widths, set_gripper

    a_skill, b_skill = chain.split('_')
    sim, block, wall, tray, info = build_scene(seed=seed, jitter=0.008)
    try:
        camera = CameraRig(sim, CAMERAS, IMAGE_SIZE, supersample=1)
        for _ in range(args.warmup_steps):
            sim.step()
        sim.allowed_contacts.update({(block, 8), (block, 9), (block, 10)})
        runner.reset()
        np.random.seed(seed)
        runner.torch.manual_seed(seed)
        runner.torch.cuda.manual_seed_all(seed)

        st = block_state(sim, block)
        a_initial = {'position': st['position'].tolist(), 'yaw': st['yaw']}
        phase = 'A'
        task = TASKS[a_skill]
        a_success = False
        b_success = False
        a_ticks = b_ticks = 0
        started = time.perf_counter()
        total_ticks = args.max_ticks_a + args.max_ticks_b
        for tick in range(total_ticks):
            if phase == 'A' and a_ticks >= args.max_ticks_a:
                break
            if phase == 'B' and b_ticks >= args.max_ticks_b:
                break
            frames = camera.capture()
            state = np.concatenate([sim.q(), np.asarray(finger_widths(sim))])
            action = runner.act(frames, state, task)
            target = np.clip(action[:7], np.asarray(sim.lower) + 1e-4,
                             np.asarray(sim.upper) - 1e-4)
            width = float(np.clip(action[7], 0.0, 0.04))
            sim.command(target)
            set_gripper(sim, width, 20.0, 0.05)
            for _ in range(STEPS_PER_ACTION):
                sim.step()
            st = block_state(sim, block)
            if phase == 'A':
                a_ticks += 1
                if a_ticks > 40 and a_done(a_skill, st):
                    a_success = True
                    phase = 'B'
                    task = TASKS[b_skill]
                    runner.reset()          # 清空旧动作块
                    yaw0 = st['yaw']
                    y0 = float(st['position'][1])
            else:
                b_ticks += 1
                done = b_done(st, yaw0, y0) if b_skill == 'B' else c_done(st)
                if b_ticks > 15 and done:
                    b_success = True
                    break
        # 判定前只让物理继续跑
        for _ in range(int(round(0.6 / sim.dt))):
            sim.step()
        st = block_state(sim, block)
        result = {
            'chain': chain, 'seed': int(seed), 'phase_reached': phase,
            'A_success': bool(a_success), 'B_success': bool(b_success),
            'chain_success': bool(a_success and b_success),
            'a_ticks': int(a_ticks), 'b_ticks': int(b_ticks),
            'wall_seconds': round(time.perf_counter() - started, 2),
            'a_initial': a_initial,
            'final_position': st['position'].tolist(),
            'final_yaw_rad': float(st['yaw']), 'final_tilt_deg': float(st['tilt']),
            'final_speed': float(st['speed']), 'final_contacts': int(st['contacts']),
        }
        return result
    finally:
        if p.isConnected(sim.client):
            p.disconnect(sim.client)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', required=True)
    ap.add_argument('--chains', nargs='+', default=['A0_B', 'A90_B', 'A0_C', 'A90_C'])
    ap.add_argument('--episodes', type=int, default=20)
    ap.add_argument('--seed-base', type=int, default=40000000)
    ap.add_argument('--max-ticks-a', type=int, default=200)
    ap.add_argument('--max-ticks-b', type=int, default=200)
    ap.add_argument('--warmup-steps', type=int, default=30)
    ap.add_argument('--n-action-steps', type=int)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runner = PolicyRunner(args.policy, args.device, args.n_action_steps)
    all_rows = []
    for chain in args.chains:
        rows = []
        for i in range(args.episodes):
            seed = args.seed_base + i
            res = run_chain(runner, chain, seed, args)
            rows.append(res)
            print(f'[{chain} {i + 1}/{args.episodes}] seed={seed} '
                  f'A={res["A_success"]} B/C={res["B_success"]} '
                  f'ticks={res["a_ticks"]}+{res["b_ticks"]}', flush=True)
        n_ok = sum(1 for r in rows if r['chain_success'])
        summary = {
            'chain': chain, 'episodes': len(rows), 'chain_success': n_ok,
            'chain_rate': round(n_ok / max(len(rows), 1), 4),
            'A_success': sum(1 for r in rows if r['A_success']),
            'conditional_B_success': sum(1 for r in rows if r['A_success'] and r['B_success']),
            'conditional_B_rate': round(
                sum(1 for r in rows if r['A_success'] and r['B_success'])
                / max(sum(1 for r in rows if r['A_success']), 1), 4),
        }
        (out / f'eval_{chain}.json').write_text(
            json.dumps({'summary': summary, 'episodes': rows}, ensure_ascii=False, indent=2),
            encoding='utf-8')
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        all_rows.extend(rows)

    fields = ['chain', 'seed', 'chain_success', 'A_success', 'B_success', 'phase_reached',
              'a_ticks', 'b_ticks', 'final_position', 'final_yaw_rad', 'final_tilt_deg',
              'final_speed', 'final_contacts', 'wall_seconds']
    with (out / 'episodes.csv').open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in all_rows:
            writer.writerow({**row, 'final_position': json.dumps(row['final_position'])})
    print('done', len(all_rows))


if __name__ == '__main__':
    main()
