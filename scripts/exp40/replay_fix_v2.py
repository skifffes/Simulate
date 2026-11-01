"""实验 4.0 Replay Fix v2：按数据集 episode 正确恢复场景，并做保真度 + 事件级阶段标注。

修正的三点（对应 review 意见）：
 1. **episode 映射**：`lerobot_panda_e23_D2` 是 480 集混合数据集（A0/A90/B90/C90 各 120 集，
    fps=20），`episode_index` 与 `b_training_start_states.csv` 的 `episode_id` **不是同一编号**。
    上一轮直接回放 `episode_index=64`（其实是别的任务）→ 推不动。本脚本用**首帧 q 与 CSV 的 q 的
    L2 距离**配对，只有 `distance < 1e-6` 才认为同一集，并用该 CSV 行恢复物体位姿。
 2. **墙体 ID**：用 `build_scene()` 返回的 `wall` 对象（`info['wall_body']`），不靠碰撞盒猜测；
    并有**正例自检**（把方块贴到墙内侧 1.5 mm，必须检出接触）。
 3. **接触/推动判据**：每个 240 Hz 物理步都查接触；`pushing` 要求接触窗口内 +X 位移 > 1 mm。
    另加**保真度**：逐帧比较重放后的关节与数据集 `observation.state`。

模式标记：`PHYSICAL_REPLAY`（真实控制 + 接触动力学）。若物体位姿只能取名义值（A90），
则标记 `APPROXIMATE_PHYSICAL_REPLAY`。

用法：
    python scripts/exp40/replay_fix_v2.py --task B90 --episode-index 241
    python scripts/exp40/replay_fix_v2.py --task A90 --episode-index 120
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

SIZE = 256
DATASET = Path('/scratch/lijh/simulate_vla/data/lerobot_panda_e23_D2')
START_CSV = ROOT / 'results' / 'experiment_3' / 'data' / 'b_training_start_states.csv'
TASK_KEYS = {'A90': 'with its long side aligned with X',
             'B90': 'push the orange block along the positive X direction to the wall'}
COLORS = {'block': (255, 90, 0), 'robot': (80, 120, 255), 'plane': (60, 60, 60),
          'wall': (200, 200, 200), 'tray': (0, 160, 160)}
# A90 名义起点：与 config2.START 一致（物体长轴沿 Y、贴在桌面上）
A90_NOMINAL_OBJ = {'pos': (0.55, 0.0, 0.021), 'quat': (0.0, 0.0, 0.0, 1.0),
                   'note': 'A90 起始物体位姿未记录在数据集内，取 config2.START 名义值'}


def sha(a) -> str:
    a = np.ascontiguousarray(a)
    return hashlib.sha256(a.tobytes() + str(a.shape).encode()).hexdigest()


def load_episode(task: str, episode_index: int | None):
    import pandas as pd

    eps = [json.loads(l) for l in (DATASET / 'meta' / 'episodes.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    info = json.loads((DATASET / 'meta' / 'info.json').read_text(encoding='utf-8'))
    task_eps = [int(e['episode_index']) for e in eps
                if any(TASK_KEYS[task] in t for t in e.get('tasks', []))]
    idx = episode_index if episode_index is not None else task_eps[0]
    if idx not in task_eps:
        raise SystemExit(f'episode {idx} 不是 {task} 任务；可用示例: {task_eps[:5]}')
    parq = DATASET / 'data' / f'chunk-{idx // 1000:03d}' / f'episode_{idx:06d}.parquet'
    df = pd.read_parquet(parq, columns=['action', 'observation.state', 'frame_index'])
    return {'episode_index': idx, 'task': task, 'fps': info.get('fps'), 'parquet': str(parq),
            'actions': np.stack(df['action'].to_numpy()).astype(float),
            'states': np.stack(df['observation.state'].to_numpy()).astype(float),
            'n_frames': int(len(df)), 'task_episodes_available': len(task_eps)}


def match_csv_row(q0: np.ndarray):
    """用首帧 q 在 B90 起始状态 CSV 里找同一集；返回 (row, distance)。"""
    if not START_CSV.exists():
        return None, float('inf')
    best, best_d = None, float('inf')
    for row in csv.DictReader(open(START_CSV, encoding='utf-8')):
        q = np.array([float(v) for v in row['q'].split(';')])
        d = float(np.linalg.norm(q - q0))
        if d < best_d:
            best, best_d = row, d
    return best, best_d


def restore(sim, block, robot_q, obj_pos, obj_quat, fingers):
    import pybullet as p

    for j, v in enumerate(robot_q):
        p.resetJointState(sim.robot, j, float(v), physicsClientId=sim.client)
    for j, v in zip((9, 10), fingers):
        p.resetJointState(sim.robot, j, float(v), physicsClientId=sim.client)
    p.resetBasePositionAndOrientation(block, list(obj_pos), list(obj_quat), physicsClientId=sim.client)
    p.resetBaseVelocity(block, [0, 0, 0], [0, 0, 0], physicsClientId=sim.client)
    sim.command(list(robot_q))
    for j, v in zip((9, 10), fingers):
        p.setJointMotorControl2(sim.robot, j, p.POSITION_CONTROL, float(v), force=30,
                                physicsClientId=sim.client)
    for _ in range(30):
        sim.step()
    pos, quat = p.getBasePositionAndOrientation(block, physicsClientId=sim.client)
    return {'object_position_error_m': float(np.linalg.norm(np.asarray(pos) - np.asarray(obj_pos))),
            'object_rotation_error_rad': float(2 * np.arccos(np.clip(abs(np.dot(quat, obj_quat)), 0, 1))),
            'robot_q_error_rad': float(np.linalg.norm(np.asarray(sim.q()) - np.asarray(robot_q))),
            'gripper_error_m': float(abs(p.getJointState(sim.robot, 9, physicsClientId=sim.client)[0] - fingers[0]))}


def colorize(seg, roles):
    img = np.zeros((*seg.shape, 3), dtype=np.uint8)
    for role, bids in roles.items():
        img[np.isin(seg, list(bids))] = COLORS.get(role, (120, 0, 120))
    return img


def seg_counts(seg, block, robot, wall, tray):
    def c(ids):
        return int(np.isin(seg, list(ids)).sum())
    def bbox(ids):
        m = np.isin(seg, list(ids))
        if not m.any():
            return None
        ys, xs = np.nonzero(m)
        return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    return {'block_px': c([block]), 'block_bbox': bbox([block]), 'block_frac': round(c([block]) / seg.size, 5),
            'robot_px': c([robot]), 'wall_px': c(wall), 'tray_px': c(tray)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', default='B90', choices=['A90', 'B90'])
    ap.add_argument('--episode-index', type=int, default=None)
    ap.add_argument('--out', default=str(ROOT / 'results' / 'experiment_4_0' / 'replay_fix' / 'v2'))
    ap.add_argument('--max-frames', type=int, default=0, help='>0 时只重放前 N 帧（调试用）')
    args = ap.parse_args(argv)

    out = Path(args.out) / args.task
    (out / 'frames').mkdir(parents=True, exist_ok=True)

    from PIL import Image
    import pybullet as p

    from scripts.exp40._common import build_sim_scene

    ep = load_episode(args.task, args.episode_index)
    q0 = ep['states'][0, :7]
    fingers0 = ep['states'][0, 7:9]

    # ---- 场景与起点 ----
    if args.task == 'B90':
        row, dist = match_csv_row(q0)
        if row is None or dist > 1e-6:
            raise SystemExit(f'未能在 start_states CSV 中找到与 episode {ep["episode_index"]} 首帧 q 匹配的行'
                             f'（最近距离 {dist:.3e}）；为避免拼错场景，中止。')
        obj_pos = [float(row['obj_x']), float(row['obj_y']), float(row['obj_z'])]
        obj_quat = [float(v) for v in row['obj_quat'].split(';')]
        provenance_obj = {'source': 'results/experiment_3/data/b_training_start_states.csv',
                          'matched_episode_id': int(row['episode_id']), 'q_distance': dist,
                          'obj_pos': obj_pos, 'obj_quat': obj_quat,
                          'obj_yaw_rad': float(row['obj_yaw_rad']), 'start_mode': row['start_mode'],
                          'episode_success_in_csv': row['episode_success']}
        replay_mode = 'PHYSICAL_REPLAY'
        seed = int(row['scene_seed'])
    else:
        obj_pos, obj_quat = list(A90_NOMINAL_OBJ['pos']), list(A90_NOMINAL_OBJ['quat'])
        provenance_obj = {'source': A90_NOMINAL_OBJ['note'], 'obj_pos': obj_pos, 'obj_quat': obj_quat}
        replay_mode = 'APPROXIMATE_PHYSICAL_REPLAY'
        seed = 0

    sim, block, rig, prompt, info = build_sim_scene(args.task, seed=seed, camera_version='v3')
    wall_body = int(info['wall_body'])
    tray_bodies = [int(b) for b in info['tray_bodies']]
    roles = {'block': [block], 'robot': [sim.robot], 'plane': [sim.plane],
             'wall': [wall_body], 'tray': tray_bodies}
    provenance = {'task': args.task, 'episode_index': ep['episode_index'], 'parquet': ep['parquet'],
                  'dataset_fps': ep['fps'], 'n_frames': ep['n_frames'],
                  'task_episodes_available': ep['task_episodes_available'],
                  'replay_mode': replay_mode, 'object_state': provenance_obj,
                  'scene': args.task, 'scene_seed': seed,
                  'wall_body': wall_body, 'tray_bodies': tray_bodies,
                  'wall_id_source': 'build_scene() 返回的 wall 对象',
                  'camera_version': camera_v2.VERSION,
                  'camera_sha256': camera_v2.CAMERA_CONFIG_SHA256,
                  'steps_per_frame': int(round(240 / float(ep['fps']))),
                  'control': 'POSITION_CONTROL 目标 = 数据集 action[:7]；夹爪目标 = action[7]'}

    try:
        errs = restore(sim, block, q0, obj_pos, obj_quat, fingers0)
        provenance['restoration_errors'] = errs
        provenance['restoration_pass'] = bool(errs['object_position_error_m'] < 2e-3
                                              and errs['object_rotation_error_rad'] < 1e-2
                                              and errs['robot_q_error_rad'] < 5e-2)

        # ---- 检测器正例自检 ----
        pos0, quat0 = p.getBasePositionAndOrientation(block, physicsClientId=sim.client)
        wall_pos = p.getBasePositionAndOrientation(wall_body, physicsClientId=sim.client)[0]
        wall_shape = p.getCollisionShapeData(wall_body, -1, physicsClientId=sim.client)[0]
        put_x = float(wall_pos[0]) - float(wall_shape[3][0]) - 0.0015
        p.resetBasePositionAndOrientation(block, [put_x, wall_pos[1], float(pos0[2])], quat0,
                                          physicsClientId=sim.client)
        p.resetBaseVelocity(block, [0, 0, 0], [0, 0, 0], physicsClientId=sim.client)
        hit = 0
        for _ in range(120):
            sim.step()
            if p.getContactPoints(bodyA=block, bodyB=wall_body, physicsClientId=sim.client):
                hit += 1
        provenance['detector_positive_control'] = {'steps_with_contact': hit, 'pass': bool(hit > 0)}
        restore(sim, block, q0, obj_pos, obj_quat, fingers0)

        # ---- 物理重放 ----
        steps_per_frame = int(round(240 / float(ep['fps'])))
        n = ep['n_frames'] if args.max_frames <= 0 else min(args.n_frames if False else ep['n_frames'],
                                                           args.max_frames)
        rows = []
        tracking = []
        first_robot_contact = first_wall_contact = None
        step_robot = step_wall = 0
        for fid in range(n):
            a = ep['actions'][fid]
            sim.command(a[:7])
            for j in (9, 10):
                p.setJointMotorControl2(sim.robot, j, p.POSITION_CONTROL, float(a[7]), force=30,
                                        physicsClientId=sim.client)
            fr = fw = 0
            for _ in range(steps_per_frame):
                sim.step()
                if p.getContactPoints(bodyA=block, bodyB=sim.robot, physicsClientId=sim.client):
                    fr += 1
                if p.getContactPoints(bodyA=block, bodyB=wall_body, physicsClientId=sim.client):
                    fw += 1
            step_robot += fr
            step_wall += fw
            pos, quat = p.getBasePositionAndOrientation(block, physicsClientId=sim.client)
            q_now = np.asarray(sim.q(), dtype=float)
            f_now = np.array([p.getJointState(sim.robot, j, physicsClientId=sim.client)[0] for j in (9, 10)])
            nxt = min(fid + 1, ep['states'].shape[0] - 1)
            tracking.append({'frame': fid,
                             'q_err_rad': float(np.linalg.norm(q_now - ep['states'][fid, :7])),
                             'q_err_next_rad': float(np.linalg.norm(q_now - ep['states'][nxt, :7])),
                             'finger_err_m': float(np.linalg.norm(f_now - ep['states'][fid, 7:9])),
                             'finger_err_next_m': float(np.linalg.norm(f_now - ep['states'][nxt, 7:9]))})
            if fr and first_robot_contact is None:
                first_robot_contact = fid
            if fw and first_wall_contact is None:
                first_wall_contact = fid
            rows.append({'frame': fid, 'block_x': round(float(pos[0]), 6),
                         'block_y': round(float(pos[1]), 6), 'block_z': round(float(pos[2]), 6),
                         'robot_contact_steps': fr, 'wall_contact_steps': fw,
                         'gripper_target_m': round(float(a[7]), 4)})
        provenance['first_robot_contact_frame'] = first_robot_contact
        provenance['first_wall_contact_frame'] = first_wall_contact
        provenance['contact_steps_total'] = {'robot': step_robot, 'wall': step_wall}
        provenance['block_displacement_m'] = {'dx': round(rows[-1]['block_x'] - obj_pos[0], 5),
                                              'dy': round(rows[-1]['block_y'] - obj_pos[1], 5)}
        te = np.array([t['q_err_rad'] for t in tracking])
        tn = np.array([t['q_err_next_rad'] for t in tracking])
        fe = np.array([t['finger_err_m'] for t in tracking])
        fn = np.array([t['finger_err_next_m'] for t in tracking])
        provenance['tracking_fidelity'] = {
            'q_err_rad_mean': float(te.mean()), 'q_err_rad_p95': float(np.percentile(te, 95)),
            'q_err_rad_max': float(te.max()),
            'q_err_next_rad_mean': float(tn.mean()), 'q_err_next_rad_p95': float(np.percentile(tn, 95)),
            'finger_err_m_mean': float(fe.mean()), 'finger_err_next_m_mean': float(fn.mean()),
            'note': '逐帧比较重放关节与数据集 observation.state；*_next 表示与下一帧 state 比较，'
                    '用于区分“真正的重放偏差”与“记录相位差一帧”'}
        provenance['tracking_fidelity']['phase_lag_hypothesis_supported'] = bool(tn.mean() < te.mean() * 0.7)

        # ---- 事件级阶段（B90 用 push/wall 判据；A90 用 grasp/lift/transport/place 判据） ----
        phases = [('initial', 0)]
        push_ok = False
        if first_robot_contact is not None:
            phases.append(('approach', max(0, first_robot_contact - 8)))
            if args.task == 'B90':
                phases.append(('first_robot_contact', first_robot_contact))
                win = rows[first_robot_contact:min(n, first_robot_contact + 12)]
                dx_win = (win[-1]['block_x'] - rows[first_robot_contact]['block_x']) if win else 0.0
                push_ok = dx_win > 0.001
                phases.append(('pushing' if push_ok else 'contact_no_forward_push',
                               min(n - 1, first_robot_contact + 6)))
            else:
                phases.append(('grasp_contact', first_robot_contact))
                z0 = obj_pos[2]
                lift_f = next((r['frame'] for r in rows if r['block_z'] > z0 + 0.02), None)
                trans_f = next((r['frame'] for r in rows if abs(r['block_y'] - obj_pos[1]) > 0.05), None)
                if lift_f is not None:
                    phases.append(('lift', lift_f))
                if trans_f is not None:
                    phases.append(('transport', trans_f))
                place_f = next((r['frame'] for r in rows
                                if abs(r['block_y'] - info['place_xy'][1]) < 0.03 and r['block_z'] < z0 + 0.03), None)
                if place_f is not None:
                    phases.append(('place_near', place_f))
        if args.task == 'B90':
            if first_wall_contact is not None:
                phases.append(('wall_contact_verified', first_wall_contact))
            else:
                phases.append(('wall_contact_not_verified', None))
        phases.append(('final', n - 1))
        phases = sorted({p[0]: p for p in phases}.values(), key=lambda kv: (kv[1] is None, kv[1] or 0))
        provenance['phases'] = [{'name': k, 'frame': v} for k, v in phases]
        provenance['pushing_criterion_met'] = bool(push_ok) if args.task == 'B90' else None

        # ---- 渲染阶段帧 ----
        vis = []
        sim.close()
        sim, block, rig, prompt, info = build_sim_scene(args.task, seed=seed, camera_version='v3')
        restore(sim, block, q0, obj_pos, obj_quat, fingers0)
        targets = {fid: name for name, fid in phases if fid is not None}
        seg_checks = []
        for fid in range(n):
            a = ep['actions'][fid]
            sim.command(a[:7])
            for j in (9, 10):
                p.setJointMotorControl2(sim.robot, j, p.POSITION_CONTROL, float(a[7]), force=30,
                                        physicsClientId=sim.client)
            for _ in range(steps_per_frame):
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
            seg_t = np.asarray(p.getCameraImage(SIZE, SIZE, view, proj, renderer=p.ER_TINY_RENDERER,
                                                flags=0, physicsClientId=sim.client)[4],
                               dtype=np.int32).reshape(SIZE, SIZE)
            seg_w = np.asarray(p.getCameraImage(SIZE, SIZE, wview, wproj, renderer=p.ER_TINY_RENDERER,
                                                flags=0, physicsClientId=sim.client)[4],
                               dtype=np.int32).reshape(SIZE, SIZE)
            pos, quat = p.getBasePositionAndOrientation(block, physicsClientId=sim.client)
            ee = np.asarray(sim.pose()[0], dtype=float)
            mt = seg_counts(seg_t, block, sim.robot, [wall_body], tray_bodies)
            mw = seg_counts(seg_w, block, sim.robot, [wall_body], tray_bodies)
            base = f'{args.task}_{name}'
            Image.fromarray(frames['top']).save(out / 'frames' / f'{base}_external.png')
            Image.fromarray(frames['wrist']).save(out / 'frames' / f'{base}_wrist.png')
            Image.fromarray(colorize(seg_t, roles)).save(out / 'frames' / f'{base}_external_segvis.png')
            Image.fromarray(colorize(seg_w, roles)).save(out / 'frames' / f'{base}_wrist_segvis.png')
            np.save(out / 'frames' / f'{base}_external_segids.npy', seg_t)
            np.save(out / 'frames' / f'{base}_wrist_segids.npy', seg_w)
            seg_checks.append({'phase': name, 'frame': fid,
                               'rgb_sha256': sha(frames['top']), 'segvis_sha256': sha(colorize(seg_t, roles)),
                               'segids_sha256': sha(seg_t),
                               'rgb_equals_segvis': sha(frames['top']) == sha(colorize(seg_t, roles)),
                               'unique_ids': [int(v) for v in np.unique(seg_t)]})
            bb = mt['block_bbox']
            vis.append({'phase': name, 'frame': fid,
                        'block_pos': [round(float(v), 5) for v in pos],
                        'ee_pos': [round(float(v), 5) for v in ee],
                        'gripper_target_m': round(float(a[7]), 4),
                        'ext_block_px': mt['block_px'], 'ext_block_frac': mt['block_frac'],
                        'ext_block_bbox': bb,
                        'ext_block_short_side_224': round(min(bb[2] - bb[0], bb[3] - bb[1]) * 224 / SIZE, 1) if bb else 0.0,
                        'ext_wall_px': mt['wall_px'], 'ext_tray_px': mt['tray_px'],
                        'ext_robot_px': mt['robot_px'],
                        'wrist_block_px': mw['block_px'], 'wrist_block_bbox': mw['block_bbox'],
                        'wrist_robot_px': mw['robot_px'],
                        'wrist_status': 'VISIBLE' if mw['block_px'] > 0 else 'TARGET_NOT_VISIBLE'})

        # ---- 落盘 ----
        (out / 'provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf-8')
        (out / 'tracking_fidelity.csv').write_text(
            'frame,q_err_rad,q_err_next_rad,finger_err_m,finger_err_next_m\n' + '\n'.join(
                f"{t['frame']},{t['q_err_rad']:.6f},{t['q_err_next_rad']:.6f},"
                f"{t['finger_err_m']:.6f},{t['finger_err_next_m']:.6f}" for t in tracking) + '\n',
            encoding='utf-8')
        with open(out / 'contact_events.csv', 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        with open(out / 'camera_visibility.csv', 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=list(vis[0]))
            w.writeheader()
            for r in vis:
                w.writerow({k: (json.dumps(r[k]) if isinstance(r[k], list) else r[k]) for k in w.fieldnames})
        (out / 'segmentation_validation.json').write_text(json.dumps(
            {'n_frames': len(seg_checks), 'checks': seg_checks,
             'all_rgb_not_equal_segvis': all(not c['rgb_equals_segvis'] for c in seg_checks)},
            ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({k: provenance[k] for k in
                          ('task', 'episode_index', 'replay_mode', 'restoration_errors',
                           'detector_positive_control', 'first_robot_contact_frame',
                           'first_wall_contact_frame', 'contact_steps_total', 'block_displacement_m',
                           'tracking_fidelity', 'pushing_criterion_met')},
                         ensure_ascii=False, indent=2))
        print('phases:', [p['name'] for p in provenance['phases']])
        print('wrist visible:', sum(1 for r in vis if r['wrist_status'] == 'VISIBLE'), '/', len(vis))
    finally:
        sim.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
