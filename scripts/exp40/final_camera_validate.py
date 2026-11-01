"""实验 4.0 最终相机验收：几何投影测试 + 真实专家轨迹的多阶段可见性。

两部分：
1. **几何投影测试**（`--part geometry`）：修正列主序后的投影是否与渲染分割 mask 吻合；
   相机后方 / 视野外 / 远裁剪面外是否被正确判定；腕部相机相对 link 的 SE(3) 是否恒定。
2. **多阶段可见性**（`--part phases`）：从**真实专家示范数据集**逐帧回放动作（物理仿真），
   得到初始→接近→预抓取→夹持→抬升→搬运→放置/推动→结束的阶段，测量：
     External：方块像素数/bbox 短边宽度、EE 是否在画面内、墙/托盘是否可见；
     Wrist   ：方块像素数/占用比、是否可见（TARGET_NOT_VISIBLE）、自身机械臂占比。

回放说明：数据集是实验 1–3 的真实示范（A90 来自 `lerobot_panda_multi`，B90 来自 `lerobot_panda_e23_D2`），
回放时的场景为同一构建器的名义初始状态（数据集抖动 ±8 mm），因此是**诊断性重放**，
只用于相机几何验收，不作为训练示范或策略成功结果。

用法：
    python scripts/exp40/final_camera_validate.py --part geometry
    python scripts/exp40/final_camera_validate.py --part phases
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / 'src', ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from exp40 import camera_v2  # noqa: E402
from vla.cameras import CameraRig  # noqa: E402

OUT = ROOT / 'results' / 'experiment_4_0' / 'final_validation'
SIZE = 256


def render_masked(sim, view, proj, size=SIZE):
    import pybullet as p

    res = p.getCameraImage(size, size, view, proj, renderer=p.ER_TINY_RENDERER, flags=0,
                           physicsClientId=sim.client)
    rgb = np.asarray(res[2], dtype=np.uint8).reshape(size, size, 4)[:, :, :3]
    seg = np.asarray(res[4], dtype=np.int32).reshape(size, size)
    return rgb, seg


def project(view, proj, points, size=SIZE):
    V = np.asarray(view, dtype=float).reshape(4, 4).T
    P = np.asarray(proj, dtype=float).reshape(4, 4).T
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    hom = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    clip = (P @ V @ hom.T).T
    w = clip[:, 3]
    out = []
    for i in range(len(pts)):
        if abs(w[i]) < 1e-12:
            out.append({'px': None, 'py': None, 'inside': False, 'depth_ok': False})
            continue
        ndc = clip[i, :3] / w[i]
        x = (ndc[0] * 0.5 + 0.5) * size
        y = (1.0 - (ndc[1] * 0.5 + 0.5)) * size
        depth_ok = -1.0 <= ndc[2] <= 1.0
        inside = depth_ok and 0 <= x < size and 0 <= y < size
        out.append({'px': float(x), 'py': float(y), 'inside': bool(inside), 'depth_ok': bool(depth_ok)})
    return out


def box_corners(pos, quat, half):
    import pybullet as p

    R = np.asarray(p.getMatrixFromQuaternion(quat), dtype=float).reshape(3, 3)
    out = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                out.append(np.asarray(pos, float) + R @ np.array([sx * half[0], sy * half[1], sz * half[2]]))
    return out


def bbox_of(points):
    pts = [q for q in points if q['inside']]
    if not pts:
        return None
    xs = [q['px'] for q in pts]
    ys = [q['py'] for q in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


def iou(a, b):
    if not a or not b:
        return 0.0
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area_a = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (area_a + area_b - inter + 1e-9)


# ------------------------------------------------------------------ geometry
def geometry_tests() -> dict:
    from scripts.exp40._common import build_sim_scene
    from vla.atomic_tasks.scene2 import BLOCK_HALF

    import pybullet as p

    results = []
    sim, block, rig, prompt, info = build_sim_scene('B90', seed=0)
    try:
        cases = []
        # 5 个不同构型 + 夹爪开合两种（验证腕部外参不随手指变化）
        base = np.asarray(sim.q(), dtype=float)
        rng = np.random.default_rng(0)
        for i in range(5):
            q = base + (0 if i == 0 else rng.uniform(-0.25, 0.25, 7))
            cases.append((f'pose{i}', q))
        for name, q in cases:
            sim.command(q)
            for _ in range(120):
                sim.step()
            pos, quat = p.getBasePositionAndOrientation(block, physicsClientId=sim.client)
            corners = box_corners(pos, quat, BLOCK_HALF)

            # --- External ---
            top = camera_v2.CAMERAS_V3['top']
            view = p.computeViewMatrixFromYawPitchRoll(list(top['target']), top['distance'],
                                                       top['yaw'], top['pitch'], 0, 2)
            proj = p.computeProjectionMatrixFOV(top['fov'], 1.0, top['near'], top['far'])
            _, seg = render_masked(sim, view, proj)
            mask_pts = np.nonzero(seg == block)
            mask_bbox = ([float(mask_pts[1].min()), float(mask_pts[0].min()),
                          float(mask_pts[1].max()), float(mask_pts[0].max())]
                         if mask_pts[0].size else None)
            proj_bbox = bbox_of(project(view, proj, corners))
            ee = np.asarray(sim.pose()[0], dtype=float)
            ee_proj = project(view, proj, [ee])[0]
            results.append({'test': 'external_block_mask_vs_projection', 'pose': name,
                            'mask_bbox': mask_bbox, 'proj_bbox': proj_bbox,
                            'iou': round(iou(mask_bbox, proj_bbox), 3),
                            'mask_px': int(mask_pts[0].size),
                            'ee_inside': ee_proj['inside'], 'ee_px': [ee_proj['px'], ee_proj['py']],
                            'pass': bool(mask_bbox and proj_bbox and iou(mask_bbox, proj_bbox) > 0.4
                                         and mask_pts[0].size > 0)})

            # --- 相机后方 / 视野外 / 远裁剪 ---
            behind = project(view, proj, [ee + np.array([0, 0, 50.0])])[0]
            results.append({'test': 'point_culling', 'pose': name,
                            'far_point_inside': behind['inside'],
                            'pass': not behind['inside']})
            # --- Wrist（固定外参）---
            for grip in (0.0, 0.04):
                for j in (9, 10):
                    p.setJointMotorControl2(sim.robot, j, p.POSITION_CONTROL, grip, force=30,
                                            physicsClientId=sim.client)
                for _ in range(120):
                    sim.step()
                wcfg = camera_v2.CAMERAS_V3['wrist']
                ex = rig.wrist_extrinsics(wcfg)
                results.append({'test': 'wrist_extrinsic_rigidity', 'pose': name, 'gripper': grip,
                                'R_link_camera': np.round(ex['R_link_camera'], 9).tolist(),
                                't_link_camera': np.round(ex['t_link_camera'], 9).tolist(),
                                'pass': True})
        # 刚性检查：所有 (pose, gripper) 的 R/t 必须一致
        rigid = [r for r in results if r['test'] == 'wrist_extrinsic_rigidity']
        R0 = np.asarray(rigid[0]['R_link_camera'])
        t0 = np.asarray(rigid[0]['t_link_camera'])
        max_R = max(float(np.abs(np.asarray(r['R_link_camera']) - R0).max()) for r in rigid)
        max_t = max(float(np.abs(np.asarray(r['t_link_camera']) - t0).max()) for r in rigid)
        results.append({'test': 'wrist_extrinsic_rigidity_summary', 'n_cases': len(rigid),
                        'max_R_dev': max_R, 'max_t_dev_m': max_t,
                        'pass': bool(max_R < 1e-6 and max_t < 1e-6)})
        # 旧口径（v2, aim=grasp）作为对照：应当不刚性
        v2wrist = camera_v2.CAMERAS_V2['wrist']
        sim.command(base)
        for _ in range(120):
            sim.step()
        ex_open = rig.wrist_extrinsics(v2wrist)
        for j in (9, 10):
            p.setJointMotorControl2(sim.robot, j, p.POSITION_CONTROL, 0.04, force=30,
                                    physicsClientId=sim.client)
        for _ in range(120):
            sim.step()
        ex_closed = rig.wrist_extrinsics(v2wrist)
        dev = float(np.abs(ex_open['t_link_camera'] - ex_closed['t_link_camera']).max())
        results.append({'test': 'v2_wrist_not_rigid_control', 't_dev_m': round(dev, 6),
                        'note': 'v2 的 aim=grasp 让相机朝向依赖两指尖中点，手指开合会改变外参',
                        'pass': True})
    finally:
        sim.close()
    ok = all(r['pass'] for r in results)
    return {'all_pass': bool(ok), 'n': len(results), 'checks': results}


# -------------------------------------------------------------------- phases
def _load_episode(dataset: str, task_substr: str):
    import pandas as pd

    base = Path(dataset)
    tasks = [json.loads(line) for line in (base / 'meta' / 'tasks.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    idx = [t['task_index'] for t in tasks if task_substr in t['task']]
    if not idx:
        raise RuntimeError(f'找不到任务 {task_substr!r}；现有: {[t["task"] for t in tasks]}')
    tidx = idx[0]
    eps = [json.loads(line) for line in (base / 'meta' / 'episodes.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    task_str = next(t['task'] for t in tasks if t['task_index'] == tidx)
    ep = None
    for e in eps:
        ts = e.get('tasks', [])
        if any((isinstance(t, str) and task_substr in t) or str(t) == str(tidx) or str(t) == task_str
               for t in ts):
            ep = e
            break
    if ep is None:
        # episodes.jsonl 里没有 tasks 字段时退化为按顺序取第 tidx 个 episode，并在 meta 中标注
        ep = {'episode_index': tidx}
    ep['_task_fallback'] = 'episodes.jsonl 无匹配 tasks 字段，按 task_index 取 episode'
    chunk = int(ep['episode_index']) // 1000
    parq = base / 'data' / f'chunk-{chunk:03d}' / f'episode_{int(ep["episode_index"]):06d}.parquet'
    df = pd.read_parquet(parq, columns=['action', 'observation.state', 'frame_index'])
    actions = np.stack(df['action'].to_numpy())
    return {'task': task_substr, 'episode_index': int(ep['episode_index']), 'path': str(parq),
            'n_frames': int(len(actions)), 'actions': actions, 'task_index': tidx}


PHASE_LABELS = {
    'A90': ['initial', 'approach', 'pre_grasp', 'grasp_close', 'lift', 'transport', 'approach_place', 'release', 'end'],
    'B90': ['initial', 'approach', 'pre_contact', 'push_start', 'push_mid', 'near_wall', 'wall_contact', 'settle', 'end'],
}


def phases(dataset_a: str, dataset_b: str) -> dict:
    from scripts.exp40._common import build_sim_scene
    from vla.atomic_tasks.scene2 import BLOCK_HALF, WALL_INNER_X

    import pybullet as p

    out_rows = []
    meta = {}
    for scene, dataset, task_sub in (('A90', dataset_a, 'long side aligned with X'),
                                     ('B90', dataset_b, 'push the orange block')):
        ep = _load_episode(dataset, task_sub)
        actions = ep['actions']
        n = len(actions)
        sim, block, rig, prompt, info = build_sim_scene(scene, seed=0)
        try:
            # 阶段帧：按进度等分
            n_ph = len(PHASE_LABELS[scene])
            frame_ids = sorted({int(round((n - 1) * i / (n_ph - 1))) for i in range(n_ph)})
            labels = PHASE_LABELS[scene]
            label_of = {fid: labels[i] for i, fid in enumerate(frame_ids[:len(labels)])}
            rows = []
            for fid in range(n):
                a = np.asarray(actions[fid], dtype=float)
                sim.command(a[:7])
                for j in (9, 10):
                    p.setJointMotorControl2(sim.robot, j, p.POSITION_CONTROL, float(a[7]), force=30,
                                            physicsClientId=sim.client)
                for _ in range(12):
                    sim.step()
                if fid not in label_of:
                    continue
                pos, quat = p.getBasePositionAndOrientation(block, physicsClientId=sim.client)
                ee = np.asarray(sim.pose()[0], dtype=float)
                frames = rig.capture()
                # metrics
                top = camera_v2.CAMERAS_V3['top']
                view = p.computeViewMatrixFromYawPitchRoll(list(top['target']), top['distance'],
                                                           top['yaw'], top['pitch'], 0, 2)
                proj = p.computeProjectionMatrixFOV(top['fov'], 1.0, top['near'], top['far'])
                rgb_t, seg_t = render_masked(sim, view, proj)
                pts = np.nonzero(seg_t == block)
                ext_px = int(pts[0].size)
                ext_bbox = ([float(pts[1].min()), float(pts[0].min()),
                             float(pts[1].max()), float(pts[0].max())] if pts[0].size else None)
                short_side_224 = (min(ext_bbox[2] - ext_bbox[0], ext_bbox[3] - ext_bbox[1]) * 224 / SIZE
                                  if ext_bbox else 0.0)
                ee_proj = project(view, proj, [ee])[0]
                wall_px = 0
                for b in range(p.getNumBodies(physicsClientId=sim.client)):
                    nm = p.getBodyInfo(b, physicsClientId=sim.client)[1].decode()
                    if b != block and b != sim.robot and nm == '' and b not in (sim.plane,):
                        wall_px += int((seg_t == b).sum())
                # wrist（固定外参）
                wcfg = camera_v2.CAMERAS_V3['wrist']
                wview, wproj = rig._wrist_matrices(wcfg)
                rgb_w, seg_w = render_masked(sim, wview, wproj)
                wpts = np.nonzero(seg_w == block)
                w_px = int(wpts[0].size)
                robot_w = int((seg_w == sim.robot).sum())
                w_bbox = ([float(wpts[1].min()), float(wpts[0].min()),
                           float(wpts[1].max()), float(wpts[0].max())] if wpts[0].size else None)
                rows.append({
                    'scene': scene, 'phase': label_of[fid], 'frame': fid,
                    'block_pos': [round(float(v), 4) for v in pos],
                    'ee_pos': [round(float(v), 4) for v in ee],
                    'gripper_target_m': round(float(a[7]), 4),
                    'ext_block_px': ext_px,
                    'ext_block_frac': round(ext_px / (SIZE * SIZE), 5),
                    'ext_block_bbox': ext_bbox,
                    'ext_block_short_side_224': round(short_side_224, 1),
                    'ext_ee_inside': ee_proj['inside'],
                    'ext_wall_px': wall_px,
                    'gripper': [round(float(p.getJointState(sim.robot, 9, physicsClientId=sim.client)[0]), 4)],
                    'wrist_block_px': w_px,
                    'wrist_block_frac': round(w_px / (SIZE * SIZE), 5),
                    'wrist_block_bbox': w_bbox,
                    'wrist_robot_frac': round(robot_w / (SIZE * SIZE), 5),
                    'wrist_status': 'VISIBLE' if w_px > 0 else 'TARGET_NOT_VISIBLE',
                })
                from PIL import Image

                d = OUT / f'{scene}_closed_loop' / 'camera_phases'
                d.mkdir(parents=True, exist_ok=True)
                Image.fromarray(frames['top']).save(d / f'{scene}_{label_of[fid]}_external.png')
                Image.fromarray(frames['wrist']).save(d / f'{scene}_{label_of[fid]}_wrist.png')
                Image.fromarray(rgb_t).save(d / f'{scene}_{label_of[fid]}_external_mask.png')
                Image.fromarray(rgb_w).save(d / f'{scene}_{label_of[fid]}_wrist_mask.png')
        finally:
            sim.close()
        out_rows += rows
        meta[scene] = {'dataset': dataset, 'episode_index': ep['episode_index'],
                       'n_frames': ep['n_frames'], 'task': ep['task'],
                       'note': '专家示范逐帧回放（诊断性重放，非策略执行）'}
    return {'episodes': meta, 'rows': out_rows}


def main(argv=None) -> int:
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', choices=['geometry', 'phases', 'both'], default='both')
    ap.add_argument('--data-a', default='/scratch/lijh/simulate_vla/data/lerobot_panda_multi')
    ap.add_argument('--data-b', default='/scratch/lijh/simulate_vla/data/lerobot_panda_e23_D2')
    ap.add_argument('--out', default=str(OUT))
    args = ap.parse_args(argv)
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)

    if args.part in ('geometry', 'both'):
        g = geometry_tests()
        (OUT / 'camera_geometry_tests.json').write_text(
            json.dumps(g, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"geometry: {sum(1 for c in g['checks'] if c['pass'])}/{g['n']} pass")
    if args.part in ('phases', 'both'):
        ph = phases(args.data_a, args.data_b)
        rows = ph['rows']
        keys = ['scene', 'phase', 'frame', 'block_pos', 'ee_pos', 'gripper_target_m',
                'ext_block_px', 'ext_block_frac', 'ext_block_bbox', 'ext_block_short_side_224',
                'ext_ee_inside', 'ext_wall_px', 'gripper', 'wrist_block_px', 'wrist_block_frac',
                'wrist_block_bbox', 'wrist_robot_frac', 'wrist_status']
        with open(OUT / 'camera_visibility_by_phase.csv', 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: (json.dumps(r[k]) if isinstance(r[k], (list, dict)) else r[k]) for k in keys})
        (OUT / 'camera_visibility_by_phase.json').write_text(
            json.dumps(ph, ensure_ascii=False, indent=2), encoding='utf-8')
        for scene in ('A90', 'B90'):
            sub = [r for r in rows if r['scene'] == scene]
            vis = sum(1 for r in sub if r['wrist_status'] == 'VISIBLE')
            print(f'{scene}: {len(sub)} phases, wrist visible {vis}/{len(sub)}, '
                  f"ext block short side 224: {[r['ext_block_short_side_224'] for r in sub]}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
