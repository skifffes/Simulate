"""实验 2 阶段 4：共享 SmolVLA 在四个原子技能上的严格自主评测。

严格口径（与实验 1.1 一致）：
- 所有动作都来自策略；判定前不替它张爪、不修正末端、不移动物体；
- 成功来自真实物体状态（位置/朝向/接触/静止）；
- 每个技能用独立测试种子，与采集种子（900000+）完全隔离。

用法：
    python scripts/eval_multi_atomic_vla.py \
        --policy outputs/smolvla_multi/checkpoints/020000/pretrained_model \
        --skills A0 A90 B C --episodes 20 --out results/experiment_2/eval_atomic
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

from vla.atomic_tasks.config2 import CAMERAS, FPS, IMAGE_SIZE, STEPS_PER_ACTION, TASKS  # noqa: E402
from vla.atomic_tasks.experts import axis_error_x, axis_error_y  # noqa: E402
from vla.atomic_tasks.scene2 import (  # noqa: E402
    BLOCK_HALF, PLACE, TRAY, WALL_INNER_X, block_yaw, build_scene,
)
from vla.cameras import CameraRig  # noqa: E402

ACTION_DIM = 8
SKILLS = ['A0', 'A90', 'B', 'C']


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
        self.calls = 0

    def reset(self):
        self.policy.reset()
        self.calls = 0

    def act(self, frames, state, task: str):
        torch = self.torch
        batch = {}
        for name in CAMERAS:
            img = np.ascontiguousarray(frames[name])
            x = torch.from_numpy(img).to(self.device).float().div_(255.0)
            batch[f'observation.images.{name}'] = x.permute(2, 0, 1).unsqueeze(0)
        batch['observation.state'] = (
            torch.from_numpy(np.asarray(state, dtype=np.float32)).to(self.device).unsqueeze(0))
        batch['task'] = [task]
        with torch.no_grad():
            action = self.policy.select_action(batch)
        self.calls += 1
        action = action.squeeze(0).float().cpu().numpy().reshape(-1)
        if action.shape[0] < ACTION_DIM:
            action = np.concatenate([action, np.zeros(ACTION_DIM - action.shape[0])])
        return action[:ACTION_DIM]


def sample_scene(skill: str, seed: int) -> dict:
    rng = np.random.default_rng(int(seed))
    if skill in ('A0', 'A90'):
        return {'jitter': 0.008, 'yaw': 0.0}
    base_yaw = 0.0 if rng.random() < 0.5 else math.pi / 2
    yaw = base_yaw + float(rng.uniform(-0.10, 0.10))
    return {'start': PLACE, 'jitter': 0.012, 'yaw': yaw}


def fingers_contact(sim, block):
    import pybullet as p
    contacts = p.getContactPoints(bodyA=block, bodyB=sim.robot, physicsClientId=sim.client)
    return sum(1 for c in contacts if c[4] in (9, 10) and c[9] > 0.0)


def evaluate(skill: str, sim, block, initial, max_z):
    import pybullet as p
    pos, quat = p.getBasePositionAndOrientation(block, physicsClientId=sim.client)
    pos = np.asarray(pos, dtype=float)
    yaw = block_yaw(block, sim.client)
    state = p.getBaseVelocity(block, physicsClientId=sim.client)
    speed = float(np.linalg.norm(state[0]))
    euler = p.getEulerFromQuaternion(quat)
    tilt = math.degrees(abs(math.atan2(math.sin(euler[0]), math.cos(euler[0])))) + \
        math.degrees(abs(math.atan2(math.sin(euler[1]), math.cos(euler[1]))))
    lift = float(max_z - initial[2])
    released = fingers_contact(sim, block) == 0
    rest = speed < 0.01
    info = {
        'final_position': pos.tolist(), 'final_yaw_rad': float(yaw),
        'final_tilt_deg': float(tilt), 'final_speed_m_s': speed,
        'lift_height_m': lift, 'released': bool(released), 'rest': bool(rest),
    }
    if skill == 'A0':
        err = float(np.linalg.norm(pos[:2] - np.asarray(PLACE[:2])))
        ok = (err <= 0.02 and axis_error_y(yaw) <= math.radians(12) and tilt <= 12
              and released and rest)
        info.update({'xy_error_m': err, 'failure': None if ok else 'place_check'})
    elif skill == 'A90':
        err = float(np.linalg.norm(pos[:2] - np.asarray(PLACE[:2])))
        ok = (err <= 0.02 and axis_error_x(yaw) <= math.radians(12) and tilt <= 12
              and released and rest)
        info.update({'xy_error_m': err, 'failure': None if ok else 'place_check'})
    elif skill == 'B':
        depth = BLOCK_HALF[0] * abs(math.cos(yaw)) + BLOCK_HALF[1] * abs(math.sin(yaw))
        gap = float(WALL_INNER_X - (pos[0] + depth))
        dy = float(pos[1] - initial[1])
        dyaw = abs(math.atan2(math.sin(yaw - initial[3]), math.cos(yaw - initial[3])))
        ok = (abs(gap) <= 0.02 and abs(dy) <= 0.02 and dyaw <= math.radians(15)
              and tilt <= 12 and rest)
        info.update({'gap_m': gap, 'lateral_dy_m': dy, 'yaw_change_rad': dyaw,
                     'failure': None if ok else 'push_check'})
    else:
        dx = float(pos[0] - TRAY[0])
        dy = float(pos[1] - TRAY[1])
        ok = (abs(dx) <= 0.02 and abs(dy) <= 0.05 and axis_error_y(yaw) <= math.radians(12)
              and tilt <= 12 and released and rest)
        info.update({'tray_dx_m': dx, 'tray_dy_m': dy,
                     'failure': None if ok else 'tray_check'})
    info['success'] = bool(ok)
    return info


def run_episode(runner, skill: str, seed: int, args):
    import pybullet as p
    from manipulation import finger_widths, set_gripper

    scene = sample_scene(skill, seed)
    sim, block, wall, tray, info = build_scene(seed=seed, **scene)
    try:
        camera = CameraRig(sim, CAMERAS, IMAGE_SIZE, supersample=args.supersample)
        for _ in range(args.warmup_steps):
            sim.step()
        initial_pos, initial_quat = p.getBasePositionAndOrientation(block,
                                                                   physicsClientId=sim.client)
        initial = (np.asarray(initial_pos, dtype=float).tolist()
                   + [block_yaw(block, sim.client)])
        max_z = float(initial_pos[2])
        runner.reset()
        np.random.seed(seed)
        runner.torch.manual_seed(seed)
        runner.torch.cuda.manual_seed_all(seed)

        latencies, width_cmds = [], []
        released_once = False
        started = time.perf_counter()
        for tick in range(args.max_ticks):
            frames = camera.capture()
            state = np.concatenate([sim.q(), np.asarray(finger_widths(sim))])
            t0 = time.perf_counter()
            action = runner.act(frames, state, TASKS[skill])
            latencies.append(time.perf_counter() - t0)
            target = np.clip(action[:7], np.asarray(sim.lower) + 1e-4,
                             np.asarray(sim.upper) - 1e-4)
            width = float(np.clip(action[7], 0.0, 0.04))
            width_cmds.append(width)
            sim.command(target)
            set_gripper(sim, width, 20.0, 0.05)
            for _ in range(STEPS_PER_ACTION):
                sim.step()
            pos, _ = p.getBasePositionAndOrientation(block, physicsClientId=sim.client)
            max_z = max(max_z, float(pos[2]))
            if tick > 20 and fingers_contact(sim, block) == 0 and width > 0.03:
                released_once = True
        # 判定前只让物理继续跑，不给任何新指令。
        for _ in range(int(round(0.6 / sim.dt))):
            sim.step()
        result = evaluate(skill, sim, block, initial, max_z)
        result.update({
            'skill': skill, 'seed': int(seed), 'ticks': tick + 1,
            'wall_seconds': round(time.perf_counter() - started, 2),
            'mean_latency_ms': float(np.mean(latencies) * 1000),
            'p95_latency_ms': float(np.percentile(latencies, 95) * 1000),
            'min_width_cmd': float(np.min(width_cmds)),
            'max_width_cmd': float(np.max(width_cmds)),
            'ever_released': bool(released_once),
            'scene': {'start': initial, 'block_start': scene.get('start', 'default'),
                      'jitter': scene['jitter'], 'yaw': scene['yaw']},
        })
        return result
    finally:
        if p.isConnected(sim.client):
            p.disconnect(sim.client)


def run_expert_episode(skill: str, seed: int, args):
    """专家基线：同一批测试种子、同一场景、同一成功判据。"""
    import pybullet as p

    from vla.atomic_tasks.experts import SkillExpert

    scene = sample_scene(skill, seed)
    sim, block, wall, tray, info = build_scene(seed=seed, **scene)
    try:
        for _ in range(args.warmup_steps):
            sim.step()
        pos, _ = p.getBasePositionAndOrientation(block, physicsClientId=sim.client)
        initial = list(pos) + [block_yaw(block, sim.client)]
        max_z = float(pos[2])
        started = time.perf_counter()
        expert = SkillExpert(sim, block)
        fn = {'A0': 'skill_a0', 'A90': 'skill_a90', 'B': 'skill_b', 'C': 'skill_c'}[skill]
        res = getattr(expert, fn)()
        pos2, _ = p.getBasePositionAndOrientation(block, physicsClientId=sim.client)
        max_z = max(max_z, float(pos2[2]))
        result = evaluate(skill, sim, block, initial, max_z)
        result.update({
            'skill': skill, 'seed': int(seed), 'ticks': len(expert.trace),
            'wall_seconds': round(time.perf_counter() - started, 2),
            'mean_latency_ms': 0.0, 'p95_latency_ms': 0.0,
            'expert_reason': res['reason'],
            'scene': {'start': initial, 'block_start': scene.get('start', 'default'),
                      'jitter': scene['jitter'], 'yaw': scene['yaw']},
        })
        return result
    finally:
        if p.isConnected(sim.client):
            p.disconnect(sim.client)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', help='checkpoint 目录（--expert 时可省略）')
    ap.add_argument('--skills', nargs='+', default=SKILLS)
    ap.add_argument('--episodes', type=int, default=20)
    ap.add_argument('--seed-base', type=int, default=40000000)
    ap.add_argument('--max-ticks', type=int, default=260)
    ap.add_argument('--warmup-steps', type=int, default=30)
    ap.add_argument('--supersample', type=int, default=1)
    ap.add_argument('--n-action-steps', type=int)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--expert', action='store_true', help='跑专家基线而不是策略')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not args.expert and not args.policy:
        raise SystemExit('非 --expert 模式需要 --policy')
    runner = None if args.expert else PolicyRunner(args.policy, args.device,
                                                   args.n_action_steps)

    all_rows = []
    for skill in args.skills:
        rows = []
        for i in range(args.episodes):
            seed = args.seed_base + i
            if args.expert:
                res = run_expert_episode(skill, seed, args)
            else:
                res = run_episode(runner, skill, seed, args)
            rows.append(res)
            print(f'[{skill} {i + 1}/{args.episodes}] seed={seed} '
                  f'{"OK  " if res["success"] else "FAIL"} '
                  f'failure={res.get("failure") or "-"} '
                  f'ticks={res["ticks"]} lift={res["lift_height_m"]:.3f}', flush=True)
        n_ok = sum(1 for r in rows if r['success'])
        summary = {
            'skill': skill, 'episodes': len(rows), 'success': n_ok,
            'success_rate': round(n_ok / max(len(rows), 1), 4),
            'mean_latency_ms': float(np.mean([r['mean_latency_ms'] for r in rows])),
            'ticks_mean': float(np.mean([r['ticks'] for r in rows])),
        }
        (out / f'eval_{skill}.json').write_text(
            json.dumps({'summary': summary, 'episodes': rows}, ensure_ascii=False, indent=2),
            encoding='utf-8')
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        all_rows.extend(rows)

    fields = ['skill', 'seed', 'success', 'failure', 'ticks', 'lift_height_m',
              'final_position', 'final_yaw_rad', 'final_tilt_deg', 'released', 'rest',
              'xy_error_m', 'gap_m', 'lateral_dy_m', 'yaw_change_rad', 'tray_dx_m',
              'tray_dy_m', 'mean_latency_ms', 'p95_latency_ms']
    with (out / 'episodes.csv').open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in all_rows:
            writer.writerow({**row, 'final_position': json.dumps(row['final_position'])})
    print('done', len(all_rows))


if __name__ == '__main__':
    main()
