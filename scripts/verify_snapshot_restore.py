"""实验 2.1 阶段 2：验证交接快照能否被准确恢复。

对每个快照：新建相同场景（同一 seed）→ restoreState(.bullet) → 对比关节、物体、
夹爪、接触与两路图像。输出 restore_report.json。

用法：
    PYTHONPATH=src venv/bin/python -u scripts/verify_snapshot_restore.py \
        --snapshots results/experiment_2_1/snapshots
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pybullet as p
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402
sys.path.insert(0, str(ROOT / 'src'))

from manipulation import finger_widths  # noqa: E402
from vla.atomic_tasks.config2 import CAMERAS, IMAGE_SIZE  # noqa: E402
from vla.atomic_tasks.scene2 import build_scene, block_yaw  # noqa: E402
from vla.cameras import CameraRig  # noqa: E402


def finger_contact_count(sim, block):
    contacts = p.getContactPoints(bodyA=block, bodyB=sim.robot, physicsClientId=sim.client)
    return sum(1 for c in contacts if c[4] in (9, 10) and c[9] > 0.0)


def read_state(sim, block):
    q = sim.q()
    pos, quat = p.getBasePositionAndOrientation(block, physicsClientId=sim.client)
    lin, ang = p.getBaseVelocity(block, physicsClientId=sim.client)
    return {
        'q': q.tolist(),
        'fingers': [float(v) for v in finger_widths(sim)],
        'obj_pos': [float(v) for v in pos],
        'obj_quat': [float(v) for v in quat],
        'obj_lin': [float(v) for v in lin],
        'obj_ang': [float(v) for v in ang],
        'contacts': finger_contact_count(sim, block),
    }


def quat_error(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    dot = float(abs(np.dot(a, b)))
    return float(2 * math.acos(min(1.0, dot)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--snapshots', default='results/experiment_2_1/snapshots')
    ap.add_argument('--out', default='results/experiment_2_1/restore_check.json')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    root = Path(args.snapshots)
    files = sorted(p for p in root.glob('*.json') if p.name != 'collection_log.json')
    if args.limit:
        files = files[:args.limit]
    reports = []
    for path in files:
        snap = json.loads(path.read_text(encoding='utf-8'))
        bullet = root / snap['bullet_file']
        sim, block, wall, tray, info = build_scene(seed=snap['seed'], jitter=0.008)
        try:
            camera = CameraRig(sim, CAMERAS, IMAGE_SIZE, supersample=1)
            before_camera = {name: np.asarray(camera.capture()[name], dtype=np.uint8).copy()
                             for name in CAMERAS}
            # 场景刚建好时的状态（未恢复）
            scene_state = read_state(sim, block)
            p.restoreState(fileName=str(bullet), physicsClientId=sim.client)
            restored = read_state(sim, block)
            after_camera = {name: np.asarray(camera.capture()[name], dtype=np.uint8).copy()
                            for name in CAMERAS}
            expected = {
                'q': snap['robot']['q'],
                'fingers': snap['robot']['finger_positions'],
                'obj_pos': snap['object']['position'],
                'obj_quat': snap['object']['quaternion'],
                'obj_lin': snap['object']['linear_velocity'],
                'obj_ang': snap['object']['angular_velocity'],
                'contacts': snap['finger_contacts'],
            }
            dq = float(np.max(np.abs(np.asarray(restored['q']) - np.asarray(expected['q']))))
            dfing = float(np.max(np.abs(np.asarray(restored['fingers'])
                                        - np.asarray(expected['fingers']))))
            dobj = float(np.linalg.norm(np.asarray(restored['obj_pos'])
                                        - np.asarray(expected['obj_pos'])))
            dquat = quat_error(restored['obj_quat'], expected['obj_quat'])
            dlin = float(np.linalg.norm(np.asarray(restored['obj_lin'])
                                        - np.asarray(expected['obj_lin'])))
            dang = float(np.linalg.norm(np.asarray(restored['obj_ang'])
                                        - np.asarray(expected['obj_ang'])))
            dcontacts = int(restored['contacts'] - expected['contacts'])
            img = {}
            for name in CAMERAS:
                saved = Image.open(root / snap[f'{name}_png'])
                saved = np.asarray(saved.convert('RGB'), dtype=np.uint8)
                img[name] = float(np.mean(np.abs(saved.astype(int) - after_camera[name].astype(int))))
            ok = (dq < 1e-4 and dfing < 1e-3 and dobj < 1e-3 and dquat < 1e-3
                  and dcontacts == 0 and max(img.values()) < 2.0)
            reports.append({
                'file': path.name, 'skill': snap['skill'], 'seed': snap['seed'],
                'joint_max_diff': dq, 'finger_max_diff': dfing,
                'object_pos_diff_m': dobj, 'object_quat_diff_rad': dquat,
                'object_linear_diff': dlin, 'object_angular_diff': dang,
                'finger_contact_diff': dcontacts,
                'image_mae': img, 'restored_ok': bool(ok),
                'scene_obj_pos': scene_state['obj_pos'],
            })
            print(f"{path.name}: dq={dq:.2e} dobj={dobj:.2e} dquat={dquat:.2e} "
                  f"contacts={dcontacts} img_mae={max(img.values()):.3f} ok={ok}", flush=True)
        finally:
            if p.isConnected(sim.client):
                p.disconnect(sim.client)

    summary = {
        'checked': len(reports),
        'passed': sum(1 for r in reports if r['restored_ok']),
        'reports': reports,
    }
    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: summary[k] for k in ('checked', 'passed')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
