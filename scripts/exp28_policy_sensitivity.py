"""实验 2.8 A：32 个 B90 中间状态的夹爪观测反事实敏感性。"""
from __future__ import annotations

import argparse, csv, json, sys
from pathlib import Path
import numpy as np
import pybullet as p
import pybullet_data
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from vla.atomic_tasks.scene2 import build_scene  # noqa: E402
from vla.atomic_tasks.config2 import CAMERAS, IMAGE_SIZE, TASKS  # noqa: E402
from vla.cameras import CameraRig  # noqa: E402

WIDTHS = [0.014, 0.020, 0.025, 0.030, 0.035, 0.040]
STEPS = [0, 4, 9, 19, 49]
INTERVALS = [(0.014, 0.025), (0.025, 0.035), (0.035, 0.040), (0.014, 0.040)]


def make_batch(policy, frames, state):
    batch = {}
    for name in CAMERAS:
        img = np.ascontiguousarray(frames[name])
        x = torch.from_numpy(img).to(policy.device).float().div_(255.0)
        batch[f'observation.images.{name}'] = x.permute(2, 0, 1).unsqueeze(0)
    batch['observation.state'] = torch.from_numpy(
        np.asarray(state, dtype=np.float32)).to(policy.device).unsqueeze(0)
    batch['task'] = [TASKS['B']]
    return batch


class FK:
    def __init__(self):
        self.c = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.c)
        self.robot = p.loadURDF('franka_panda/panda.urdf', useFixedBase=True,
                                physicsClientId=self.c)

    def xyz(self, qs):
        out = []
        for q in qs:
            for j in range(7):
                p.resetJointState(self.robot, j, float(q[j]), physicsClientId=self.c)
            out.append(p.getLinkState(self.robot, 11, computeForwardKinematics=True,
                                      physicsClientId=self.c)[4])
        return np.asarray(out)

    def close(self):
        p.disconnect(self.c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--policy', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(args.manifest.open(encoding='utf-8-sig')))
    rows = [r for i, r in enumerate(rows) if i % args.nshards == args.shard]
    policy = SmolVLAPolicy.from_pretrained(args.policy)
    policy.to('cuda'); policy.eval(); policy.device = next(policy.parameters()).device
    fk = FK()
    preds = []
    sens = []
    for r in rows:
        sid = r['state_id']
        sim, block, wall, tray, info = build_scene()
        try:
            p.restoreState(fileName=str(ROOT / r['bullet']), physicsClientId=sim.client)
            rig = CameraRig(sim, CAMERAS, IMAGE_SIZE, supersample=1)
            frames = rig.capture()
            q = np.asarray(sim.q(), float)
        finally:
            p.disconnect(sim.client)
        chunks = []
        zs = []
        for w in WIDTHS:
            cw, zw = [], []
            for seed in range(args.seeds):
                state = np.concatenate([q, [w, w]])
                policy.reset()
                torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                with torch.no_grad():
                    chunk = policy.predict_action_chunk(make_batch(policy, frames, state))
                chunk = chunk.squeeze(0).float().cpu().numpy()
                if chunk.shape[0] < 50:
                    raise SystemExit(f'chunk too short {chunk.shape}')
                cw.append(chunk); zw.append(fk.xyz(chunk[:, :7]))
            chunks.append(np.asarray(cw)); zs.append(np.asarray(zw))
        chunks = np.asarray(chunks); zs = np.asarray(zs)
        np.savez_compressed(args.out / f'{sid}.npz', chunks=chunks, z=zs[:, :, :, 2], xyz=zs,
                            q=q, grip=float(r['grip_obs_m']), seed_count=args.seeds)
        for wi, w in enumerate(WIDTHS):
            for seed in range(args.seeds):
                for k in range(50):
                    preds.append({'state_id': sid, 'seed': seed,
                                  'gripper_mm': round(w * 1000),
                                  'chunk_step': k,
                                  **{f'q{j+1}': float(chunks[wi, seed, k, j]) for j in range(7)},
                                  'ee_x': float(zs[wi, seed, k, 0]),
                                  'ee_y': float(zs[wi, seed, k, 1]),
                                  'ee_z': float(zs[wi, seed, k, 2]),
                                  'delta_z_mm': float((zs[wi, seed, k, 2] - zs[0, seed, k, 2]) * 1000),
                                  'gripper_target': float(chunks[wi, seed, k, 7])})
        for lo, hi in INTERVALS:
            li = WIDTHS.index(lo); hi_i = WIDTHS.index(hi)
            for k in STEPS:
                dz = (zs[hi_i, :, k, 2] - zs[li, :, k, 2]) * 1000
                sens.append({'state_id': sid, 'episode_id': r['episode_id'],
                             'frame_id': r['frame_id'], 'stage': r['stage_progress_label'],
                             'progress': r['episode_progress'],
                             'intervention': f'{int(lo*1000)}-{int(hi*1000)}mm',
                             'g_low_mm': int(lo * 1000), 'g_high_mm': int(hi * 1000),
                             'chunk_step': k,
                             'mean_delta_z_mm': float(dz.mean()),
                             'median_delta_z_mm': float(np.median(dz)),
                             'sensitivity_mm_per_mm': float(dz.mean() / (hi - lo) / 1000),
                             'std_delta_z_mm': float(dz.std()),
                             'num_seeds': args.seeds,
                             'positive_fraction': float((dz > 0).mean())})
        print('done', sid, flush=True)
    with (args.out / f'predictions_{args.shard}.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(preds[0].keys())); w.writeheader(); w.writerows(preds)
    with (args.out / f'sensitivity_{args.shard}.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(sens[0].keys())); w.writeheader(); w.writerows(sens)
    fk.close()


if __name__ == '__main__':
    main()
