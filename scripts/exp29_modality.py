"""实验 2.9 A：物理一致的图像/本体夹爪交叉干预。"""
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
from vla.atomic_tasks.scene2 import build_scene  # noqa: E402
from vla.atomic_tasks.config2 import CAMERAS, IMAGE_SIZE, TASKS  # noqa: E402
from vla.cameras import CameraRig  # noqa: E402
from manipulation import finger_widths, set_gripper  # noqa: E402

CONDS = [('P14-V14', 0.014, 0.014), ('P35-V35', 0.035, 0.035),
         ('P40-V40', 0.040, 0.040), ('P14-V40', 0.014, 0.040),
         ('P40-V14', 0.040, 0.014)]
HORIZONS = [0, 4, 9, 19, 49]


class FK:
    def __init__(self):
        self.c = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.c)
        self.r = p.loadURDF('franka_panda/panda.urdf', useFixedBase=True, physicsClientId=self.c)
    def xyz(self, qs):
        out = []
        for q in qs:
            for j in range(7): p.resetJointState(self.r, j, float(q[j]), physicsClientId=self.c)
            out.append(p.getLinkState(self.r, 11, computeForwardKinematics=True, physicsClientId=self.c)[4])
        return np.asarray(out)
    def close(self): p.disconnect(self.c)


def batch(policy, frames, state):
    b = {}
    for n in CAMERAS:
        x = torch.from_numpy(np.ascontiguousarray(frames[n])).to(policy.device).float().div_(255.0)
        b[f'observation.images.{n}'] = x.permute(2, 0, 1).unsqueeze(0)
    b['observation.state'] = torch.from_numpy(np.asarray(state, np.float32)).to(policy.device).unsqueeze(0)
    b['task'] = [TASKS['B']]
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--policy', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seeds', type=int, default=10)
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    states = list(csv.DictReader(args.manifest.open(encoding='utf-8-sig')))
    policy = SmolVLAPolicy.from_pretrained(args.policy); policy.to('cuda'); policy.eval()
    policy.device = next(policy.parameters()).device
    fk = FK(); preds = []; mod = []; sens = []
    for s in states:
        sim, block, wall, tray, info = build_scene(seed=int(s['seed']), start=(0.5, 0, 0), jitter=0, yaw=0.0)
        try:
            p.restoreState(fileName=str(ROOT / s['bullet']), physicsClientId=sim.client)
            q = np.asarray(sim.q(), float)
            frames = {}
            for pw in (0.014, 0.035, 0.040):
                for j in (9, 10): p.resetJointState(sim.robot, j, pw, physicsClientId=sim.client)
                sim.command(q); set_gripper(sim, pw, 20.0, 0.05)
                for _ in range(10): sim.step()
                frames[pw] = CameraRig(sim, CAMERAS, IMAGE_SIZE, supersample=1).capture()
        finally:
            p.disconnect(sim.client)
        chunks = []; zs = []
        for cname, pw, vw in CONDS:
            cw = []; zw = []
            for seed in range(args.seeds):
                st = np.concatenate([q, [vw, vw]])
                policy.reset(); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                with torch.no_grad():
                    ch = policy.predict_action_chunk(batch(policy, frames[pw], st)).squeeze(0).float().cpu().numpy()
                cw.append(ch); zw.append(fk.xyz(ch[:, :7]))
            chunks.append(np.asarray(cw)); zs.append(np.asarray(zw))
        chunks = np.asarray(chunks); zs = np.asarray(zs)
        np.savez_compressed(args.out / f"{s['state_id']}.npz", chunks=chunks, xyz=zs, q=q,
                            condition=[c[0] for c in CONDS], seed_count=args.seeds)
        for ci, (cname, pw, vw) in enumerate(CONDS):
            for seed in range(args.seeds):
                for k in range(50):
                    preds.append({'state_id': s['state_id'], 'kind': s['kind'], 'condition': cname,
                                  'image_width_mm': round(pw * 1000), 'body_width_mm': round(vw * 1000),
                                  'seed': seed, 'chunk_step': k,
                                  'ee_x': float(zs[ci, seed, k, 0]), 'ee_y': float(zs[ci, seed, k, 1]),
                                  'ee_z': float(zs[ci, seed, k, 2]),
                                  'gripper_target': float(chunks[ci, seed, k, 7])})
        for k in HORIZONS:
            for lo, hi, tag in ((0.014, 0.035, '14-35'), (0.035, 0.040, '35-40'),
                                (0.014, 0.040, '14-40')):
                li = [c[0] for c in CONDS].index(f'P{int(lo*1000)}-V{int(lo*1000)}')
                hi_i = [c[0] for c in CONDS].index(f'P{int(hi*1000)}-V{int(hi*1000)}')
                dz = (zs[hi_i, :, k, 2] - zs[li, :, k, 2]) * 1000
                sens.append({'state_id': s['state_id'], 'kind': s['kind'], 'intervention': tag,
                             'chunk_step': k, 'mean_delta_z_mm': float(dz.mean()),
                             'std_delta_z_mm': float(dz.std()), 'positive_fraction': float((dz > 0).mean()),
                             'num_seeds': args.seeds})
        for ci, (cname, pw, vw) in enumerate(CONDS):
            for k in HORIZONS:
                arr = zs[ci, :, k, 2] * 1000
                mod.append({'state_id': s['state_id'], 'kind': s['kind'], 'condition': cname,
                            'image_width_mm': round(pw * 1000), 'body_width_mm': round(vw * 1000),
                            'chunk_step': k, 'mean_z_mm': float(arr.mean()), 'std_z_mm': float(arr.std()),
                            'n_seeds': args.seeds})
        print('done', s['state_id'], flush=True)
    with (args.out / 'policy_predictions.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(preds[0].keys())); w.writeheader(); w.writerows(preds)
    with (args.out / 'modality_interventions.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(mod[0].keys())); w.writeheader(); w.writerows(mod)
    with (args.out / 'policy_height_sensitivity.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(sens[0].keys())); w.writeheader(); w.writerows(sens)
    fk.close()


if __name__ == '__main__':
    main()
