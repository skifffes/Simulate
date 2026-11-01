"""Hotfix A 收尾：冻结相机的前后对照（外部 + 腕部，多阶段，224×224 实际模型输入）。

对每个阶段渲染：
  before = 实验 2/3/4.0 基线相机（camera_v2.CAMERAS_V1_BASELINE）
  after  = 冻结相机（camera_v2.CAMERAS_V2）
并给出分割 mask 的客观指标（物体/机器人像素占比、bbox、是否在画面内）。

阶段状态来源（如实标注）：
  * `home`：场景初始化后的真实状态；
  * `trace_tN`：归档的 π₀.₅ 真实闭环轨迹第 N 个控制步的 q（诊断性重放，不是编造的专家过程）。

用法：
    python scripts/exp40/hotfix_camera_final.py
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
from exp40.observation_adapter import resize_with_pad  # noqa: E402
from scripts.exp40._common import build_sim_scene, save_json  # noqa: E402
from scripts.exp40.hotfix_camera_audit import (  # noqa: E402
    RAW_SIZE, metrics_for, load_trace_ticks, render_with_mask,
)

OUT = ROOT / 'results' / 'experiment_4_0' / 'hotfix'


def render_camera(sim, cameras, name):
    from vla.cameras import CameraRig

    rig = CameraRig(sim, cameras=cameras, image_size=(RAW_SIZE, RAW_SIZE), supersample=1)
    frames = rig.capture()
    return frames[name]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(OUT / 'camera_before_after'))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from PIL import Image
    import pybullet as p

    rows = []
    for scene in ('A90', 'B90'):
        sim, block, rig, prompt, info = build_sim_scene(scene, seed=0)
        try:
            bodies = {'block': block, 'robot': sim.robot, 'plane': sim.plane}
            stages = [{'name': 'home', 'tick': -1, 'q': list(sim.q())}]
            stages += [{'name': f'trace_t{s["tick"]}', 'tick': s['tick'], 'q': s['q']}
                       for s in load_trace_ticks(scene, 4)]
            for st in stages:
                sim.command(np.array(st['q']))
                for _ in range(40):
                    sim.step()
                for ver, cameras in (('before', camera_v2.CAMERAS_V1_BASELINE),
                                     ('after', camera_v2.CAMERAS_V2)):
                    for cam in ('top', 'wrist'):
                        rgb256 = render_camera(sim, cameras, cam)
                        img224 = resize_with_pad(rgb256, 224, 224)
                        base = f'{scene}_{cam}_{ver}_{st["name"]}'
                        Image.fromarray(rgb256).save(out / f'{base}_raw256.png')
                        Image.fromarray(img224).save(out / f'{base}_model224.png')
                        # 指标：只对固定外部相机算分割 mask；腕部是随动相机，另存图像供人工/几何核对
                        if cam == 'top':
                            _, seg, _ = render_with_mask(sim, cameras[cam])
                            m = metrics_for(seg, bodies)
                        else:
                            m = {'block_px': None, 'block_frac': None, 'robot_px': None,
                                 'robot_frac': None, 'block_bbox': None, 'robot_bbox': None}
                        rows.append({'scene': scene, 'camera': cam, 'version': ver,
                                     'stage': st['name'], 'tick': st['tick'],
                                     'model_input_hw': '224x224',
                                     'image_sha256_224': _sha(Image.fromarray(img224).tobytes()),
                                     **{k: v for k, v in m.items() if k.endswith(('_frac', '_px', '_bbox'))}})
            save_json(out / 'camera_config_used.json',
                      {'version': camera_v2.VERSION,
                       'sha256': camera_v2.CAMERA_CONFIG_SHA256,
                       'cameras_v2': camera_v2.CAMERAS_V2,
                       'cameras_v1_baseline': camera_v2.CAMERAS_V1_BASELINE})
        finally:
            sim.close()

    keys = ['scene', 'camera', 'version', 'stage', 'tick', 'model_input_hw',
            'block_px', 'block_frac', 'robot_px', 'robot_frac', 'block_bbox', 'robot_bbox',
            'image_sha256_224']
    path = out / 'camera_before_after_metrics.csv'
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(r.get(k)) if isinstance(r.get(k), list) else r.get(k)) for k in keys})
    # 汇总
    print(f'rows={len(rows)} -> {path}')
    for scene in ('A90', 'B90'):
        for cam in ('top', 'wrist'):
            for ver in ('before', 'after'):
                sub = [r for r in rows if r['scene'] == scene and r['camera'] == cam and r['version'] == ver]
                if not sub:
                    continue
                bf = np.mean([r['block_frac'] for r in sub])
                rf = np.mean([r['robot_frac'] for r in sub])
                seen = np.mean([1.0 if r['block_px'] > 0 else 0.0 for r in sub])
                print(f'{scene} {cam:5s} {ver:6s} block_frac={bf:.4f} robot_frac={rf:.4f} block_visible_frac={seen:.2f}')
    return 0


def _sha(b: bytes) -> str:
    import hashlib

    return hashlib.sha256(b).hexdigest()


def _wrist_as_fixed(sim, cameras):
    """腕部相机是随动相机，`render_with_mask` 只接受固定参数。

    为了让 mask 与腕部视图一致，这里用 CameraRig 的内部接口拿到当前腕部的 (view, proj)，
    再换算成等价的固定相机描述（仅用于渲染 mask，不改变真实安装）。
    """
    from vla.cameras import CameraRig

    rig = CameraRig(sim, cameras=cameras, image_size=(RAW_SIZE, RAW_SIZE), supersample=1)
    view, proj = rig._wrist_matrices(cameras['wrist'])
    # 用 view 矩阵反解 eye / target / up，交给 computeViewMatrix
    V = np.asarray(view, dtype=float).reshape(4, 4).T
    R = V[:3, :3]
    eye = -R.T @ V[:3, 3]
    forward = -R[2, :]
    up = R[1, :]
    target = eye + forward * 0.3
    return {'type': 'fixed', 'eye': eye.tolist(), 'target': target.tolist(), 'up': up.tolist(),
            'fov': cameras['wrist']['fov'], 'near': cameras['wrist']['near'], 'far': cameras['wrist']['far']}


if __name__ == '__main__':
    raise SystemExit(main())
