"""实验 4.0 Replay Fix：在**正确的 B90 episode 场景**中重放专家动作并重新审计相机。

背景（本轮已确认的缺陷）：
  * 上一轮 `final_camera_validate.py --part phases` 用 `build_sim_scene('B90', seed=0)` 建场景，
    方块落在 `(0.55, 0, 0.02)`、yaw=0；而实验 3 的 B90 训练起点状态是
    `obj ≈ (0.55, 0.18, 0.02)`、`obj_yaw ≈ 90°`（见 `results/experiment_3/data/b_training_start_states.csv`）。
    => 机械臂沿真实 B90 轨迹运动，方块却在另一个位置，腕部相机自然看不到目标。
  * 同一脚本把 **RGB 当作 mask** 保存（`*_external_mask.png` 与 RGB 同源），mask 归档无效。

本脚本做三件事：
  1. 从 `b_training_start_states.csv` 恢复该 episode 的**方块位姿 + 机器人 q + 夹爪**，并逐项核对误差；
  2. 用数据集里的动作序列做**物理重放**（`sim.command(q_target)`，物体靠接触动力学响应），
     逐帧检测真实接触事件（机器人↔方块、方块↔墙），阶段标签只来自这些事件；
  3. 保存 RGB、**彩色可视化 mask** 与**原始 int32 seg id**（.npy），二者可独立复核。

用法：
    python scripts/exp40/replay_fix_validate.py --episode-id 64
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / 'src', ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from exp40 import camera_v2  # noqa: E402

OUT = ROOT / 'results' / 'experiment_4_0' / 'replay_fix'
SIZE = 256
START_CSV = ROOT / 'results' / 'experiment_3' / 'data' / 'b_training_start_states.csv'
DATASET = Path('/scratch/lijh/simulate_vla/data/lerobot_panda_e23_D2')

# mask 可视化配色（与 body 角色绑定，不是随机色）
COLORS = {'block': (255, 90, 0), 'robot': (80, 120, 255), 'plane': (60, 60, 60),
          'wall': (200, 200, 200), 'tray': (0, 160, 160), 'other': (120, 0, 120)}


def sha(a) -> str:
    a = np.ascontiguousarray(a)
    return hashlib.sha256(a.tobytes() + str(a.shape).encode()).hexdigest()


def load_start_state(episode_id: int) -> dict | None:
    with open(START_CSV, encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            if int(row['episode_id']) == episode_id:
                q = [float(v) for v in row['q'].split(';')]
                oq = [float(v) for v in row['obj_quat'].split(';')]
                return {'episode_id': episode_id, 'q': q, 'obj_pos': [float(row['obj_x']), float(row['obj_y']), float(row['obj_z'])],
                        'obj_quat': oq, 'obj_yaw': float(row['obj_yaw_rad']),
                        'fingers': [float(row['finger_left']), float(row['finger_right'])],
                        'scene_seed': int(row['scene_seed']), 'yaw_label': row['yaw'],
                        'start_mode': row['start_mode'], 'episode_success': row['episode_success']}
    return None


def load_actions(episode_id: int):
    import pandas as pd

    base = DATASET
    eps = [json.loads(l) for l in (base / 'meta' / 'episodes.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    ep = next((e for e in eps if int(e['episode_index']) == episode_id), None)
    if ep is None:
        ep = eps[0]
    idx = int(ep['episode_index'])
    chunk = idx // 1000
    parq = base / 'data' / f'chunk-{chunk:03d}' / f'episode_{idx:06d}.parquet'
    df = pd.read_parquet(parq, columns=['action', 'observation.state', 'frame_index'])
    return idx, str(parq), np.stack(df['action'].to_numpy()), np.stack(df['observation.state'].to_numpy())


def restore(sim, block, st: dict) -> dict:
    import pybullet as p

    # 方块：位置 + 四元数（xyzw）+ 清零速度
    p.resetBasePositionAndOrientation(block, st['obj_pos'], st['obj_quat'], physicsClientId=sim.client)
    p.resetBaseVelocity(block, [0, 0, 0], [0, 0, 0], physicsClientId=sim.client)
    # 机器人：q + 夹爪
    for j, v in enumerate(st['q']):
        p.resetJointState(sim.robot, j, float(v), physicsClientId=sim.client)
    for j, v in zip((9, 10), st['fingers']):
        p.resetJointState(sim.robot, j, float(v), physicsClientId=sim.client)
    sim.command(st['q'])
    for j, v in zip((9, 10), st['fingers']):
        p.setJointMotorControl2(sim.robot, j, p.POSITION_CONTROL, float(v), force=30,
                                physicsClientId=sim.client)
    for _ in range(30):
        sim.step()
    pos, quat = p.getBasePositionAndOrientation(block, physicsClientId=sim.client)
    q = np.asarray(sim.q(), dtype=float)
    ang = lambda a, b: float(2 * np.arccos(np.clip(abs(np.dot(a, b)), 0, 1)))
    return {'object_position_error_m': float(np.linalg.norm(np.asarray(pos) - np.asarray(st['obj_pos']))),
            'object_rotation_error_rad': ang(np.asarray(quat), np.asarray(st['obj_quat'])),
            'robot_q_error_rad': float(np.linalg.norm(q - np.asarray(st['q']))),
            'gripper_error_m': float(abs(p.getJointState(sim.robot, 9, physicsClientId=sim.client)[0] - st['fingers'][0]))}


def seg_metrics(seg, block, robot, wall_bodies, tray_bodies):
    def m(bids):
        mask = np.isin(seg, list(bids))
        return int(mask.sum()), (np.nonzero(mask) if mask.any() else None)

    out = {}
    n, nz = m([block]); out['block_px'] = n
    out['block_bbox'] = [int(nz[1].min()), int(nz[0].min()), int(nz[1].max()), int(nz[0].max())] if nz else None
    out['block_frac'] = round(n / seg.size, 5)
    out['robot_px'] = m([robot])[0]
    out['wall_px'] = m(wall_bodies)[0]
    out['tray_px'] = m(tray_bodies)[0]
    return out


def colorize(seg, roles: dict) -> np.ndarray:
    img = np.zeros((*seg.shape, 3), dtype=np.uint8)
    for role, bids in roles.items():
        col = COLORS.get(role, COLORS['other'])
        img[np.isin(seg, list(bids))] = col
    return img


def main(argv=None) -> int:
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument('--episode-id', type=int, default=64)
    ap.add_argument('--scene', default='B90', choices=['B90', 'A90'])
    ap.add_argument('--out', default=str(OUT))
    args = ap.parse_args(argv)
    OUT = Path(args.out)
    (OUT / 'camera_corrected_frames').mkdir(parents=True, exist_ok=True)

    from PIL import Image
    import pybullet as p

    from scripts.exp40._common import build_sim_scene

    st = load_start_state(args.episode_id)
    idx, parq, actions, states = load_actions(args.episode_id)
    sim, block, rig, prompt, info = build_sim_scene(args.scene, seed=st['scene_seed'] if st else 0,
                                                    camera_version='v3')
    provenance = {'scene': args.scene, 'requested_episode_id': args.episode_id,
                  'dataset_episode_index': idx, 'parquet': parq,
                  'start_state_csv': str(START_CSV.relative_to(ROOT)), 'start_state_found': bool(st),
                  'camera_version': camera_v2.VERSION,
                  'camera_sha256': camera_v2.CAMERA_CONFIG_SHA256,
                  'replay_mode': 'PHYSICAL_REPLAY (sim.command 关节位置目标 + 接触动力学；'
                                 '动作语义 = 数据集动作序列，未做额外缩放)',
                  'wall_bodies': None}
    try:
        if st is None:
            print('WARN: 未找到该 episode 的起始状态，退化为名义场景（结果标记为 invalid）')
            st = {'q': list(sim.q()), 'obj_pos': [0.55, 0.0, 0.021], 'obj_quat': [0, 0, 0, 1],
                  'fingers': [0.035, 0.035], 'scene_seed': 0, 'yaw_label': 'unknown',
                  'start_mode': 'unknown', 'episode_success': 'unknown'}
        errs = restore(sim, block, st)
        provenance['restoration_errors'] = errs
        provenance['restoration_pass'] = bool(errs['object_position_error_m'] < 2e-3
                                              and errs['object_rotation_error_rad'] < 1e-2
                                              and errs['robot_q_error_rad'] < 5e-2)
        # 场景 body 角色：**直接使用 build_scene 返回的权威 ID**（wall/tray），
        # 不再用碰撞盒尺寸猜测（上一版把墙误判成托盘，导致 wall_bodies=[] 空跑）
        wall_bodies = [int(info['wall_body'])]
        tray_bodies = [int(b) for b in info['tray_bodies']]
        provenance['wall_id_source'] = 'build_scene() 返回值（info["wall_body"]）'
        provenance['wall_bodies'] = wall_bodies
        provenance['tray_bodies'] = tray_bodies
        roles = {'block': [block], 'robot': [sim.robot], 'plane': [sim.plane],
                 'wall': wall_bodies, 'tray': tray_bodies}
        provenance['roles'] = {k: list(map(int, v)) for k, v in roles.items()}

        # ---- 检测器正例自检：把方块放到墙前 3 mm，推进 120 步，探测器必须报接触 ----
        import pybullet as p2
        pos0, quat0 = p2.getBasePositionAndOrientation(block, physicsClientId=sim.client)
        wall_shape = p2.getCollisionShapeData(wall_bodies[0], -1, physicsClientId=sim.client)[0]
        wall_pos = p2.getBasePositionAndOrientation(wall_bodies[0], physicsClientId=sim.client)[0]
        put_x = float(wall_pos[0]) - float(wall_shape[3][0]) - 0.0015  # 墙内侧 x - 方块半宽
        p2.resetBasePositionAndOrientation(block, [put_x, wall_pos[1], float(pos0[2])], quat0,
                                           physicsClientId=sim.client)
        p2.resetBaseVelocity(block, [0, 0, 0], [0, 0, 0], physicsClientId=sim.client)
        hit = 0
        for _ in range(120):
            sim.step()
            if p2.getContactPoints(bodyA=block, bodyB=wall_bodies[0], physicsClientId=sim.client):
                hit += 1
        provenance['detector_positive_control'] = {'steps_with_contact': hit, 'pass': hit > 0}
        # 复位到 episode 起始状态
        restore(sim, block, st)

        rows = []
        first_robot_contact = None
        first_wall_contact = None
        prev_x = st['obj_pos'][0]
        step_robot_contacts = 0   # 逐物理步计数（帧末只读一次会漏掉瞬断接触）
        step_wall_contacts = 0
        for fid in range(len(actions)):
            a = np.asarray(actions[fid], dtype=float)
            sim.command(a[:7])
            for j in (9, 10):
                p.setJointMotorControl2(sim.robot, j, p.POSITION_CONTROL, float(a[7]), force=30,
                                        physicsClientId=sim.client)
            frame_rc = frame_wc = 0
            for _ in range(12):
                sim.step()
                # 严格接触判定：每个物理步都查（含瞬断接触）
                if p.getContactPoints(bodyA=block, bodyB=sim.robot, physicsClientId=sim.client):
                    frame_rc += 1
                for wb in wall_bodies:
                    if p.getContactPoints(bodyA=block, bodyB=wb, physicsClientId=sim.client):
                        frame_wc += 1
                        break
            step_robot_contacts += frame_rc
            step_wall_contacts += frame_wc
            pos, quat = p.getBasePositionAndOrientation(block, physicsClientId=sim.client)
            if frame_rc and first_robot_contact is None:
                first_robot_contact = fid
            if frame_wc and first_wall_contact is None:
                first_wall_contact = fid
            rows.append({'frame': fid, 'block_pos': [round(float(v), 5) for v in pos],
                         'block_x': round(float(pos[0]), 5), 'dx': round(float(pos[0]) - prev_x, 6),
                         'robot_contact_steps': frame_rc, 'wall_contact_steps': frame_wc,
                         'gripper_target_m': round(float(a[7]), 4)})
            prev_x = float(pos[0])
        provenance['first_robot_contact_frame'] = first_robot_contact
        provenance['first_wall_contact_frame'] = first_wall_contact
        provenance['final_block_pos'] = rows[-1]['block_pos']
        provenance['block_total_dx_m'] = round(rows[-1]['block_x'] - st['obj_pos'][0], 5)

        # ---- 阶段标注：只依据真实事件 ----
        n = len(rows)
        phases = [('initial', 0)]
        push_ok = False
        if first_robot_contact is not None:
            phases.append(('approach', max(0, first_robot_contact - 8)))
            phases.append(('first_robot_contact', first_robot_contact))
            # “pushing”只有在接触期间出现**实际 +X 位移**时才成立，否则只标 contact_no_forward_push
            win = rows[first_robot_contact:min(n, first_robot_contact + 12)]
            dx_win = (win[-1]['block_x'] - rows[first_robot_contact]['block_x']) if win else 0.0
            push_ok = dx_win > 0.001
            phases.append(('pushing' if push_ok else 'contact_no_forward_push',
                           min(n - 1, first_robot_contact + 6)))
        if first_wall_contact is not None:
            phases.append(('wall_contact_verified', first_wall_contact))
        else:
            phases.append(('wall_contact_not_verified', None))
        phases.append(('final', n - 1))
        provenance['phase_source'] = '机器人↔方块 / 方块↔墙 的逐物理步接触记录；' \
                                     'pushing 需接触窗口内 +X 位移 > 1 mm'
        provenance['contact_steps_total'] = {'robot': step_robot_contacts, 'wall': step_wall_contacts}
        provenance['pushing_criterion_met'] = bool(push_ok)

        # ---- 渲染（RGB + 可视化 mask + 原始 seg id） ----
        # 重新构建一次场景并在目标帧渲染：物理重放必须从头按序执行，不能跳帧
        vis_rows = []
        sim.close()
        sim, block, rig, prompt, info = build_sim_scene(args.scene, seed=st['scene_seed'] if st else 0,
                                                        camera_version='v3')
        restore(sim, block, st)
        targets = {fid: name for name, fid in phases if fid is not None}
        seg_checks = []
        for fid in range(len(actions)):
            a = np.asarray(actions[fid], dtype=float)
            sim.command(a[:7])
            for j in (9, 10):
                p.setJointMotorControl2(sim.robot, j, p.POSITION_CONTROL, float(a[7]), force=30,
                                        physicsClientId=sim.client)
            for _ in range(12):
                sim.step()
            if fid not in targets:
                continue
            name = targets[fid]
            frames = rig.capture()
            top = camera_v2.CAMERAS_V3['top']
            view = p.computeViewMatrixFromYawPitchRoll(list(top['target']), top['distance'],
                                                       top['yaw'], top['pitch'], 0, 2)
            proj = p.computeProjectionMatrixFOV(top['fov'], 1.0, top['near'], top['far'])
            wview, wproj = rig._wrist_matrices(camera_v2.CAMERAS_V3['wrist'])
            res_t = p.getCameraImage(SIZE, SIZE, view, proj, renderer=p.ER_TINY_RENDERER, flags=0,
                                     physicsClientId=sim.client)
            res_w = p.getCameraImage(SIZE, SIZE, wview, wproj, renderer=p.ER_TINY_RENDERER, flags=0,
                                     physicsClientId=sim.client)
            seg_t = np.asarray(res_t[4], dtype=np.int32).reshape(SIZE, SIZE)
            seg_w = np.asarray(res_w[4], dtype=np.int32).reshape(SIZE, SIZE)
            ee = np.asarray(sim.pose()[0], dtype=float)
            pos, quat = p.getBasePositionAndOrientation(block, physicsClientId=sim.client)
            mt = seg_metrics(seg_t, block, sim.robot, wall_bodies, tray_bodies)
            mw = seg_metrics(seg_w, block, sim.robot, wall_bodies, tray_bodies)
            base = f'{args.scene}_{name}'
            Image.fromarray(frames['top']).save(OUT / 'camera_corrected_frames' / f'{base}_external.png')
            Image.fromarray(frames['wrist']).save(OUT / 'camera_corrected_frames' / f'{base}_wrist.png')
            Image.fromarray(colorize(seg_t, roles)).save(OUT / 'camera_corrected_frames' / f'{base}_external_segvis.png')
            Image.fromarray(colorize(seg_w, roles)).save(OUT / 'camera_corrected_frames' / f'{base}_wrist_segvis.png')
            np.save(OUT / 'camera_corrected_frames' / f'{base}_external_segids.npy', seg_t)
            np.save(OUT / 'camera_corrected_frames' / f'{base}_wrist_segids.npy', seg_w)
            seg_checks.append({'phase': name, 'frame': fid,
                               'rgb_sha256': sha(frames['top']),
                               'segvis_sha256': sha(colorize(seg_t, roles)),
                               'segids_sha256': sha(seg_t),
                               'rgb_equals_segvis': sha(frames['top']) == sha(colorize(seg_t, roles)),
                               'unique_ids': [int(v) for v in np.unique(seg_t)][:8]})
            vis_rows.append({'scene': args.scene, 'phase': name, 'frame': fid,
                             'block_pos': [round(float(v), 5) for v in pos],
                             'ee_pos': [round(float(v), 5) for v in ee],
                             'gripper_target_m': round(float(a[7]), 4),
                             'ext_block_px': mt['block_px'], 'ext_block_frac': mt['block_frac'],
                             'ext_block_bbox': mt['block_bbox'],
                             'ext_block_short_side_224': round(min(mt['block_bbox'][2] - mt['block_bbox'][0],
                                                                   mt['block_bbox'][3] - mt['block_bbox'][1]) * 224 / SIZE, 1)
                             if mt['block_bbox'] else 0.0,
                             'ext_robot_px': mt['robot_px'], 'ext_wall_px': mt['wall_px'],
                             'wrist_block_px': mw['block_px'], 'wrist_block_frac': mw['block_frac'],
                             'wrist_block_bbox': mw['block_bbox'], 'wrist_robot_px': mw['robot_px'],
                             'wrist_status': 'VISIBLE' if mw['block_px'] > 0 else 'TARGET_NOT_VISIBLE'})
        (OUT / 'replay_provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf-8')
        (OUT / 'episode_restoration_checks.json').write_text(json.dumps(
            {'episode_id': args.episode_id, 'dataset_episode_index': idx,
             'errors': provenance['restoration_errors'],
             'thresholds': {'object_position_error_m': 2e-3, 'object_rotation_error_rad': 1e-2,
                            'robot_q_error_rad': 5e-2},
             'pass': provenance['restoration_pass'],
             'start_state': st}, ensure_ascii=False, indent=2), encoding='utf-8')
        (OUT / 'segmentation_validation.json').write_text(json.dumps(
            {'n_frames': len(seg_checks), 'checks': seg_checks,
             'note': 'mask 来自 PyBullet segmentation buffer（bodyUniqueId，int32）；'
                     '另存彩色可视化（segvis）与原始 id（segids.npy），RGB 与 segvis 的 sha256 必须不同',
             'all_rgb_not_equal_segvis': all(not c['rgb_equals_segvis'] for c in seg_checks)},
            ensure_ascii=False, indent=2), encoding='utf-8')
        keys = ['scene', 'phase', 'frame', 'block_pos', 'ee_pos', 'gripper_target_m', 'ext_block_px',
                'ext_block_frac', 'ext_block_bbox', 'ext_block_short_side_224', 'ext_robot_px',
                'ext_wall_px', 'wrist_block_px', 'wrist_block_frac', 'wrist_block_bbox',
                'wrist_robot_px', 'wrist_status']
        with open(OUT / 'camera_visibility_corrected.csv', 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in vis_rows:
                w.writerow({k: (json.dumps(r[k]) if isinstance(r[k], list) else r[k]) for k in keys})
        with open(OUT / 'contact_phase_validation.csv', 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=['frame', 'block_x', 'dx', 'robot_contact_steps',
                                               'wall_contact_steps', 'gripper_target_m'])
            w.writeheader()
            for r in rows:
                w.writerow({k: r[k] for k in w.fieldnames})
        print(json.dumps({'restoration_errors': provenance['restoration_errors'],
                          'first_robot_contact_frame': first_robot_contact,
                          'first_wall_contact_frame': first_wall_contact,
                          'total_dx_m': provenance['block_total_dx_m'],
                          'phases': [p[0] for p in phases],
                          'wrist_visible': sum(1 for r in vis_rows if r['wrist_status'] == 'VISIBLE'),
                          'n_phases': len(vis_rows)}, ensure_ascii=False, indent=2))
    finally:
        sim.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
