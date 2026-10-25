"""实验 2.1 阶段 5：同一交接状态下，改变语言指令对首个动作块的影响。

对若干真实 A 终态快照：
  - 用正确的后继指令（B 或 C）生成一个 chunk；
  - 用前序 A 的指令（A0/A90）生成一个 chunk；
比较夹爪开合、末端位移方向与抬升意图。

用法：
    PYTHONPATH=src venv/bin/python -u scripts/probe_language_conditioning.py \
        --snapshots results/experiment_2_1/snapshots --states 4 \
        --out results/experiment_2_1/language
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pybullet as p

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from eval_multi_atomic_vla import PolicyRunner  # noqa: E402
from manipulation import finger_widths  # noqa: E402
from vla.atomic_tasks.config2 import CAMERAS, IMAGE_SIZE, TASKS  # noqa: E402
from vla.atomic_tasks.scene2 import build_scene  # noqa: E402
from vla.cameras import CameraRig  # noqa: E402


def ee_pose_for(sim, q):
    for j, value in enumerate(q[:7]):
        p.resetJointState(sim.robot, j, float(value), 0.0, physicsClientId=sim.client)
    for k, joint in enumerate((9, 10)):
        p.resetJointState(sim.robot, joint, float(np.clip(q[7], 0.0, 0.04)), 0.0,
                          physicsClientId=sim.client)
    return sim.pose()[0]


def chunk_metrics(targets, ee_positions, grip):
    targets = np.asarray(targets)
    ee = np.asarray(ee_positions)
    return {
        'gripper_first': float(grip[0]),
        'gripper_mean': float(np.mean(grip)),
        'gripper_min': float(np.min(grip)),
        'closing_steps': int(np.sum(np.asarray(grip) < 0.02)),
        'ee_dz': float(ee[-1][2] - ee[0][2]),
        'ee_dxy': float(np.linalg.norm(ee[-1][:2] - ee[0][:2])),
        'joint_delta_norm': float(np.linalg.norm(targets[-1][:7] - targets[0][:7])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', default='outputs/smolvla_multi/checkpoints/020000/pretrained_model')
    ap.add_argument('--snapshots', default='results/experiment_2_1/snapshots')
    ap.add_argument('--states', type=int, default=4, help='每个 A 技能取几个快照')
    ap.add_argument('--horizon', type=int, default=50)
    ap.add_argument('--out', default='results/experiment_2_1/language')
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runner = PolicyRunner(args.policy)

    records = []
    for a_skill in ('A0', 'A90'):
        snaps = sorted(Path(args.snapshots).glob(f'{a_skill}_*.json'))[:args.states]
        for path in snaps:
            snap = json.loads(path.read_text(encoding='utf-8'))
            sim, block, wall, tray, info = build_scene(seed=snap['seed'], jitter=0.008)
            try:
                camera = CameraRig(sim, CAMERAS, IMAGE_SIZE, supersample=1)
                p.restoreState(fileName=str(Path(args.snapshots) / snap['bullet_file']),
                               physicsClientId=sim.client)
                sim.command(np.asarray(snap['last_action'][:7], dtype=float))
                frames = camera.capture()
                state = np.concatenate([sim.q(), np.asarray(finger_widths(sim))])
                conditions = {'B': TASKS['B'], 'C': TASKS['C'], 'A': TASKS[a_skill]}
                row = {'skill': a_skill, 'seed': snap['seed'], 'chunks': {}}
                for tag, task in conditions.items():
                    runner.reset()
                    np.random.seed(snap['seed'])
                    runner.torch.manual_seed(snap['seed'])
                    runner.torch.cuda.manual_seed_all(snap['seed'])
                    targets, grips, ees = [], [], []
                    for _ in range(args.horizon):
                        action = runner.act(frames, state, task)
                        targets.append(action[:7])
                        grips.append(float(np.clip(action[7], 0.0, 0.04)))
                        ees.append(ee_pose_for(sim, np.concatenate([action[:7], [grips[-1]]])))
                    runner.policy.reset()
                    row['chunks'][tag] = chunk_metrics(targets, ees, grips)
                records.append(row)
                print(json.dumps({'skill': a_skill, 'seed': snap['seed'],
                                  'B': row['chunks']['B'], 'A': row['chunks']['A']},
                                 ensure_ascii=False), flush=True)
            finally:
                if p.isConnected(sim.client):
                    p.disconnect(sim.client)

    (out / 'language_probe.json').write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')

    # 汇总：B 指令 vs A 指令的差异
    summary = []
    for row in records:
        b, a = row['chunks']['B'], row['chunks']['A']
        summary.append({
            'skill': row['skill'], 'seed': row['seed'],
            'B_closing_steps': b['closing_steps'], 'A_closing_steps': a['closing_steps'],
            'B_gripper_mean': b['gripper_mean'], 'A_gripper_mean': a['gripper_mean'],
            'B_ee_dz': b['ee_dz'], 'A_ee_dz': a['ee_dz'],
            'B_ee_dxy': b['ee_dxy'], 'A_ee_dxy': a['ee_dxy'],
            'delta_gripper': b['gripper_mean'] - a['gripper_mean'],
            'delta_dz': b['ee_dz'] - a['ee_dz'],
        })
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                      encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
