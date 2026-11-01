"""Hotfix A：相机几何审计与可见性度量（A90/B90 多阶段）。

做三件事：
1. 打印当前相机几何（外参近似：eye/target/up、fov、near/far、渲染分辨率）；
2. 用**归档的 π₀.₅ 闭环轨迹真实 q**（不是编造的专家过程）重放若干阶段，渲染 RGB + 分割 mask；
3. 对候选外部相机算客观可见性指标（物体的像素占比、bbox、机器人占比、目标区域是否在画面内）。

分割 mask 来自 PyBullet ER_SEGMENTATION_MASK：每个像素给出 bodyUniqueId，
因此“可见性”是真实像素计数，不是几何代理。

用法：
    python scripts/exp40/hotfix_camera_audit.py [--out ...] [--tag before|candidates]
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

from scripts.exp40._common import build_sim_scene, save_json  # noqa: E402

RAW_SIZE = 256
MODEL_HW = (224, 224)

# ---- 候选外部相机（第三人称固定视角，物理上合理、可复现） -------------------
CANDIDATES = {
    'current': {'target': (0.55, 0.21, 0.02), 'distance': 1.05, 'yaw': 50.0, 'pitch': -52.0,
                'fov': 50.0, 'near': 0.02, 'far': 5.0},
    'c1_closer': {'target': (0.58, 0.16, 0.06), 'distance': 0.72, 'yaw': 45.0, 'pitch': -38.0,
                  'fov': 55.0, 'near': 0.02, 'far': 5.0},
    'c2_oblique': {'target': (0.60, 0.16, 0.05), 'distance': 0.80, 'yaw': 35.0, 'pitch': -42.0,
                   'fov': 50.0, 'near': 0.02, 'far': 5.0},
    'c3_overhead': {'target': (0.58, 0.14, 0.05), 'distance': 0.65, 'yaw': 55.0, 'pitch': -32.0,
                    'fov': 60.0, 'near': 0.02, 'far': 5.0},
}


def camera_geometry(sim, cfg) -> dict:
    import pybullet as p

    view = p.computeViewMatrixFromYawPitchRoll(
        list(cfg['target']), cfg['distance'], cfg['yaw'], cfg['pitch'], 0, 2)
    proj = p.computeProjectionMatrixFOV(
        cfg['fov'], RAW_SIZE / RAW_SIZE, cfg['near'], cfg['far'])
    # 由 view 矩阵反解 eye / right / up / forward（行优先 4x4 -> 3x4）
    V = np.asarray(view, dtype=float).reshape(4, 4).T
    R = V[:3, :3]
    eye = -R.T @ V[:3, 3]
    forward = -R[2, :]
    return {'view': [v for row in np.asarray(view).reshape(4, 4).tolist() for v in row],
            'eye': eye.round(6).tolist(),
            'forward': forward.round(6).tolist(),
            'target': list(cfg['target']), 'distance': cfg['distance'],
            'yaw_deg': cfg['yaw'], 'pitch_deg': cfg['pitch'], 'fov_deg': cfg['fov'],
            'near': cfg['near'], 'far': cfg['far'],
            'raw_render_hw': [RAW_SIZE, RAW_SIZE], 'model_input_hw': list(MODEL_HW)}


def render_with_mask(sim, cfg):
    """返回 (rgb HxWx3 uint8, seg HxW int, depth HxW float)。"""
    import pybullet as p

    view = p.computeViewMatrixFromYawPitchRoll(
        list(cfg['target']), cfg['distance'], cfg['yaw'], cfg['pitch'], 0, 2)
    proj = p.computeProjectionMatrixFOV(cfg['fov'], 1.0, cfg['near'], cfg['far'])
    w = h = RAW_SIZE
    res = p.getCameraImage(w, h, view, proj, renderer=p.ER_TINY_RENDERER,
                           flags=0, physicsClientId=sim.client)  # flags=0 即保留 segmentation buffer
    rgb = np.asarray(res[2], dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
    depth = np.asarray(res[3], dtype=np.float32).reshape(h, w)
    seg = np.asarray(res[4], dtype=np.int32).reshape(h, w)
    return rgb, seg, depth


def project_point(sim, cfg, point):
    """把世界点投到像素坐标（用于核对“投影与分割一致”）。"""
    import pybullet as p

    view = np.asarray(p.computeViewMatrixFromYawPitchRoll(
        list(cfg['target']), cfg['distance'], cfg['yaw'], cfg['pitch'], 0, 2), dtype=float).reshape(4, 4)
    proj = np.asarray(p.computeProjectionMatrixFOV(cfg['fov'], 1.0, cfg['near'], cfg['far']),
                      dtype=float).reshape(4, 4)
    v = np.array([*point, 1.0])
    clip = proj @ view @ v
    if abs(clip[3]) < 1e-9:
        return None
    ndc = clip[:3] / clip[3]
    if ndc[2] < -1 or ndc[2] > 1:
        return None
    x = (ndc[0] * 0.5 + 0.5) * RAW_SIZE
    y = (1 - (ndc[1] * 0.5 + 0.5)) * RAW_SIZE
    inside = 0 <= x < RAW_SIZE and 0 <= y < RAW_SIZE
    return {'px': float(x), 'py': float(y), 'inside': bool(inside)}


def metrics_for(seg, bodies: dict) -> dict:
    total = seg.size
    out = {'seg_available': bool(seg.max() > 0)}
    for name, bid in bodies.items():
        cnt = int((seg == bid).sum())
        out[f'{name}_px'] = cnt
        out[f'{name}_frac'] = round(cnt / total, 5)
        if cnt > 0:
            ys, xs = np.nonzero(seg == bid)
            out[f'{name}_bbox'] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        else:
            out[f'{name}_bbox'] = None
    rb = out.get('robot_bbox')
    blk = out.get('block_bbox')
    if rb and blk:
        # 物体是否被机器人像素覆盖（粗略遮挡代理：bbox 交叠比例）
        ix = max(0, min(rb[2], blk[2]) - max(rb[0], blk[0]))
        iy = max(0, min(rb[3], blk[3]) - max(rb[1], blk[1]))
        inter = ix * iy
        area_b = max(1, (blk[2] - blk[0] + 1) * (blk[3] - blk[1] + 1))
        out['block_robot_bbox_overlap_frac'] = round(inter / area_b, 4)
    return out


def projected_box_metrics(sim, cfg, block, half):
    """几何投影代理：把方块 8 个角点投影到图像，给出 bbox 与面积占比。

    与真实像素可见性（分割 mask）的区别：投影不考虑遮挡，也不含光照/抗锯齿，
    因此它是“几何可见性”的**上界**。分割 mask 可用时以 mask 为准。
    """
    import pybullet as p

    pos, quat = p.getBasePositionAndOrientation(block, physicsClientId=sim.client)
    R = np.asarray(p.getMatrixFromQuaternion(quat), dtype=float).reshape(3, 3)
    corners = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                local = np.array([sx * half[0], sy * half[1], sz * half[2]])
                corners.append(np.asarray(pos, dtype=float) + R @ local)
    pts = [project_point(sim, cfg, c) for c in corners]
    inside = [q for q in pts if q and q['inside']]
    if not inside:
        return {'proj_block_inside': False, 'proj_block_area_frac': 0.0, 'proj_block_bbox': None}
    xs = [q['px'] for q in inside]
    ys = [q['py'] for q in inside]
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    return {'proj_block_inside': True,
            'proj_block_area_frac': round((w * h) / (RAW_SIZE * RAW_SIZE), 5),
            'proj_block_bbox': [round(min(xs), 1), round(min(ys), 1), round(max(xs), 1), round(max(ys), 1)]}


def load_trace_ticks(scene: str, n_stages: int = 4):
    """从归档的 π₀.₅ 闭环 trace 里取若干 tick 的真实 q（真实执行记录，不是编造过程）。

    注意：归档的 `trace_*.csv` 表头把 7 个关节写成了一列名字 `q1..q7`，
    而数据行是 7 个独立字段（表头字段数 < 数据字段数），因此必须**按位置**解析，
    不能用 DictReader（用名字会错位）。这一点记在 hotfix_report 的已知缺陷里。
    """
    path = ROOT / 'results' / 'experiment_4_0' / 'closed_loop_smoke' / f'trace_{scene}_seed0.csv'
    if not path.exists():
        return []
    rows = []
    with open(path, encoding='utf-8') as fh:
        rd = csv.reader(fh)
        next(rd, None)  # 表头
        for row in rd:
            if len(row) < 9 or row[1] != 'execute':
                continue
            rows.append({'tick': int(row[0]), 'q': [float(v) for v in row[2:9]]})
    if not rows:
        return []
    idxs = np.linspace(0, len(rows) - 1, n_stages).round().astype(int)
    return [rows[i] for i in sorted(set(idxs.tolist()))]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / 'results' / 'experiment_4_0' / 'hotfix' / 'camera_before_after'))
    ap.add_argument('--scenes', default='A90,B90')
    ap.add_argument('--modes', default='external,wrist')
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    geometry = {}
    from PIL import Image

    for scene in args.scenes.split(','):
        sim, block, rig, prompt, info = build_sim_scene(scene, seed=0)
        try:
            bodies = {'block': block, 'robot': sim.robot, 'plane': sim.plane}
            import pybullet as p

            # 墙/托盘按几何名识别：取所有 body 里除上述之外的
            all_bodies = [b for b in range(p.getNumBodies(physicsClientId=sim.client))]
            for b in all_bodies:
                if b in bodies.values():
                    continue
                n = p.getBodyInfo(b, physicsClientId=sim.client)[1].decode()
                bodies[f'body{b}_{n or "unnamed"}'] = b

            stages = [{'name': 'home', 'tick': -1, 'q': sim.q().tolist()}]
            stages += [{'name': f'trace_t{s["tick"]}', 'tick': s['tick'], 'q': s['q']}
                       for s in load_trace_ticks(scene, 4)]

            for cand_name, cfg in CANDIDATES.items():
                geometry[f'{scene}/external/{cand_name}'] = camera_geometry(sim, cfg)
                for st in stages:
                    sim.command(np.array(st['q']))
                    for _ in range(40):
                        sim.step()
                    rgb, seg, depth = render_with_mask(sim, cfg)
                    m = metrics_for(seg, bodies)
                    from vla.atomic_tasks.scene2 import BLOCK_HALF
                    m.update(projected_box_metrics(sim, cfg, block, BLOCK_HALF))
                    # 关键点投影核对
                    pos, quat = p.getBasePositionAndOrientation(block, physicsClientId=sim.client)
                    ee, ee_q = sim.pose()
                    proj_blk = project_point(sim, cfg, list(pos))
                    proj_ee = project_point(sim, cfg, list(ee))
                    row = {'scene': scene, 'camera': 'external', 'candidate': cand_name,
                           'stage': st['name'], 'tick': st['tick'],
                           'block_proj': proj_blk, 'ee_proj': proj_ee,
                           'block_pos': [round(float(v), 4) for v in pos],
                           'ee_pos': [round(float(v), 4) for v in ee],
                           **m}
                    rows.append(row)
                    Image.fromarray(rgb).save(out / f'{scene}_external_{cand_name}_{st["name"]}_raw{RAW_SIZE}.png')
                    Image.fromarray(rgb).resize(MODEL_HW, Image.BILINEAR).save(
                        out / f'{scene}_external_{cand_name}_{st["name"]}_model224.png')
            save_json(out / 'camera_geometry.json', geometry)
        finally:
            sim.close()

    # CSV
    keys = ['scene', 'camera', 'candidate', 'stage', 'tick', 'seg_available', 'block_px', 'block_frac',
            'robot_px', 'robot_frac', 'block_bbox', 'robot_bbox', 'block_robot_bbox_overlap_frac',
            'proj_block_inside', 'proj_block_area_frac', 'proj_block_bbox',
            'block_proj', 'ee_proj', 'block_pos', 'ee_pos']
    csv_path = args.out + '/camera_visibility_metrics.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(r.get(k), ensure_ascii=False) if isinstance(r.get(k), (dict, list)) else r.get(k))
                        for k in keys})
    print(f'rows={len(rows)} -> {csv_path}')
    # 简要汇总
    for scene in args.scenes.split(','):
        print(f'--- {scene} ---')
        for cand in CANDIDATES:
            sub = [r for r in rows if r['scene'] == scene and r['candidate'] == cand]
            if not sub:
                continue
            bf = np.mean([r['block_frac'] for r in sub])
            rf = np.mean([r['robot_frac'] for r in sub])
            pa = np.mean([r.get('proj_block_area_frac', 0.0) for r in sub])
            vis = np.mean([1.0 if (r.get('block_proj') or {}).get('inside') else 0.0 for r in sub])
            print(f'  {cand:12s} block_frac={bf:.4f} proj_area={pa:.4f} robot_frac={rf:.4f} block_in_view={vis:.2f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
