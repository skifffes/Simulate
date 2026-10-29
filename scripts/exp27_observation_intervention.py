"""实验 2.7 实验 1：固定图像/七关节，仅改夹爪本体观测的离线反事实推理。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from manipulation import finger_widths, set_gripper  # noqa: E402
from vla.atomic_tasks.scene2 import build_scene  # noqa: E402
from vla.cameras import CameraRig  # noqa: E402
from vla.atomic_tasks.config2 import CAMERAS, IMAGE_SIZE, TASKS  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402

WIDTHS = [0.014, 0.020, 0.025, 0.030, 0.035, 0.040]


def fk_z(qs):
    c = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=c)
    robot = p.loadURDF('franka_panda/panda.urdf', useFixedBase=True, physicsClientId=c)
    out = []
    for q in qs:
        for j in range(7):
            p.resetJointState(robot, j, float(q[j]), physicsClientId=c)
        out.append(p.getLinkState(robot, 11, computeForwardKinematics=True,
                                  physicsClientId=c)[4][2])
    p.disconnect(c)
    return np.asarray(out)


def make_batch(policy, frames, state, task):
    batch = {}
    for name in CAMERAS:
        img = np.ascontiguousarray(frames[name])
        x = torch.from_numpy(img).to(policy.device).float().div_(255.0)
        batch[f'observation.images.{name}'] = x.permute(2, 0, 1).unsqueeze(0)
    batch['observation.state'] = torch.from_numpy(
        np.asarray(state, dtype=np.float32)).to(policy.device).unsqueeze(0)
    batch['task'] = [task]
    return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--snapshot', type=Path, required=True)
    ap.add_argument('--policy', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seeds', type=int, default=20)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'paired_action_chunks').mkdir(exist_ok=True)
    snap = json.loads(args.snapshot.read_text())
    policy = SmolVLAPolicy.from_pretrained(args.policy)
    policy.to('cuda'); policy.eval()
    policy.device = next(policy.parameters()).device
    sets = {}
    for set_name, width in (('real40', 0.040), ('real14', 0.014)):
        sim, block, wall, tray, info = build_scene()
        try:
            p.restoreState(fileName=str(ROOT / snap['bullet']), physicsClientId=sim.client)
            for j in (9, 10):
                p.resetJointState(sim.robot, j, width, physicsClientId=sim.client)
            sim.command(sim.q())
            set_gripper(sim, width, 20.0, 0.05)
            for _ in range(10):
                sim.step()
            rig = CameraRig(sim, CAMERAS, IMAGE_SIZE, supersample=1)
            frames = rig.capture()
            q = np.asarray(sim.q(), dtype=float)
            sets[set_name] = {'frames': frames, 'q': q, 'width': width,
                              'obj': p.getBasePositionAndOrientation(block, physicsClientId=sim.client)[0],
                              'hash': hashlib.sha1(np.asarray(frames['top']).tobytes()).hexdigest()[:12]}
        finally:
            p.disconnect(sim.client)
    manifest = []
    stats = []
    for set_name, data in sets.items():
        for width in WIDTHS:
            chunks = []
            for seed in range(args.seeds):
                state = np.concatenate([data['q'], [width, width]])
                batch = make_batch(policy, data['frames'], state, TASKS['B'])
                policy.reset()
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                with torch.no_grad():
                    chunk = policy.predict_action_chunk(batch).squeeze(0).float().cpu().numpy()
                chunks.append(chunk)
                np.savez_compressed(args.out / 'paired_action_chunks' /
                                   f'{set_name}_w{int(width*1000):03d}_s{seed:02d}.npz',
                                   chunk=chunk, q=data['q'], width=width, seed=seed,
                                   set=set_name, top=data['frames']['top'],
                                   wrist=data['frames']['wrist'])
                manifest.append({'set': set_name, 'width_m': width, 'seed': seed,
                                 'q': data['q'].tolist(), 'image_hash': data['hash'],
                                 'chunk_file': str(args.out / 'paired_action_chunks' /
                                                   f'{set_name}_w{int(width*1000):03d}_s{seed:02d}.npz')})
            chunks = np.asarray(chunks)
            # action differences vs 14mm at each seed (paired)
            if width != 0.014:
                base = []
                for seed in range(args.seeds):
                    pth = args.out / 'paired_action_chunks' / f'{set_name}_w014_s{seed:02d}.npz'
                    base.append(np.load(pth)['chunk'])
                base = np.asarray(base)
                dq = np.linalg.norm(chunks[:, :10, :7] - base[:, :10, :7], axis=2).mean(axis=1)
            else:
                dq = np.zeros(args.seeds)
            zs = np.asarray([fk_z(c[:, :7]) for c in chunks])
            stats.append({'set': set_name, 'width_m': width,
                          'joint_l2_mean': float(dq.mean()), 'joint_l2_std': float(dq.std()),
                          'ee_z_t0': float(zs[:, 0].mean()),
                          'ee_z_t5': float(zs[:, 5].mean()),
                          'ee_z_t10': float(zs[:, 10].mean()),
                          'ee_z_t20': float(zs[:, 20].mean()),
                          'ee_z_t49': float(zs[:, 49].mean()),
                          'dz_first10': float((zs[:, 10] - zs[:, 0]).mean()),
                          'dz_first10_std': float((zs[:, 10] - zs[:, 0]).std()),
                          'gripper_target_mean': float(chunks[:, 0, 7].mean())})
            print(set_name, width, 'dq', round(float(dq.mean()), 4),
                  'dz10', round(float((zs[:, 10] - zs[:, 0]).mean()), 5), flush=True)
    with (args.out / 'input_manifest.csv').open('w', newline='', encoding='utf-8-sig') as f:
        import csv
        w = csv.DictWriter(f, fieldnames=['set', 'width_m', 'seed', 'q', 'image_hash', 'chunk_file'])
        w.writeheader()
        for r in manifest:
            r = dict(r); r['q'] = ';'.join(f'{v:.6f}' for v in r['q']); w.writerow(r)
    with (args.out / 'action_statistics.csv').open('w', newline='', encoding='utf-8-sig') as f:
        import csv
        w = csv.DictWriter(f, fieldnames=list(stats[0].keys())); w.writeheader(); w.writerows(stats)
    (args.out / 'analysis.json').write_text(json.dumps({'snapshot': str(args.snapshot),
                                                        'seeds': args.seeds,
                                                        'stats': stats}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
