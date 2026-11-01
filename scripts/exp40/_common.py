"""实验 4.0 脚本共用工具：场景、观测、策略客户端、工件写出。

只做“胶水”，不隐藏任何约定：所有阈值/键名都从 exp40 / vla 的常量模块取。
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT / 'src', ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from exp40 import droid_contract as dc  # noqa: E402

OUT_ROOT = ROOT / 'results' / 'experiment_4_0'


def out_dir(name: str) -> pathlib.Path:
    d = OUT_ROOT / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(path, obj) -> pathlib.Path:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, default=float)
    return path


def sha256_bytes(arr: bytes) -> str:
    return hashlib.sha256(arr).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def build_sim_scene(kind: str, seed: int | None = None, start=None, yaw: float = 0.0,
                    camera_version: str = 'v3'):
    """返回 (sim, block, rig, prompt, info)。

    kind = 'B90' 或 'A90'：两者都用实验 2/3 的同一场景构建器（scene2.build_scene），
    区别只在初始状态与语言指令，不改变任务定义。

    相机（Hotfix 3）：实验 4.0 必须使用**显式版本化**的相机配置，不允许静默退回
    `config2.CAMERAS`。`camera_version='v3'` 为当前冻结版本（外部 v2 + 腕部固定外参），
    `'v1'` 仅用于 before/after 对照，版本与 SHA256 会写进 info 随工件落盘。
    """
    from vla.cameras import CameraRig

    from exp40 import camera_v2

    if kind not in ('A90', 'B90'):
        raise ValueError(f'未知场景 {kind}')
    versions = {'v1': camera_v2.CAMERAS_V1_BASELINE,
                'v2': camera_v2.CAMERAS_V2,
                'v3': camera_v2.CAMERAS_V3}
    if camera_version not in versions:
        raise ValueError(f'未知相机版本 {camera_version}；可用: {sorted(versions)}')
    cameras = versions[camera_version]
    sha = camera_v2.config_sha256(cameras)

    from vla.atomic_tasks.scene2 import PLACE, build_scene

    sim, block, wall, tray, info = build_scene(seed=seed, jitter=0.0, start=start, yaw=yaw)
    prompt = dc.PROMPT_B90 if kind == 'B90' else _a90_prompt()
    rig = CameraRig(sim, cameras=cameras, image_size=(256, 256), supersample=1)
    info = dict(info)
    info['scene'] = kind
    info['place_xy'] = list(PLACE[:2])
    info['camera_version'] = camera_version
    info['camera_sha256'] = sha
    info['cameras'] = cameras
    return sim, block, rig, prompt, info


def _a90_prompt() -> str:
    from vla.atomic_tasks import config2

    return config2.TASKS['A90']


def capture(sim, rig) -> dict:
    """采集一次观测（不推进物理）。"""
    import pybullet as p

    frames = rig.capture()
    q = sim.q()
    fingers = np.array([p.getJointState(sim.robot, 9, physicsClientId=sim.client)[0],
                        p.getJointState(sim.robot, 10, physicsClientId=sim.client)[0]])
    ee_pos, ee_quat = sim.pose()
    return {'frames': frames, 'q': q, 'fingers': fingers,
            'ee_position': ee_pos, 'ee_quat': ee_quat}


def make_policy_client(host: str, port: int):
    """π₀.₅ websocket 客户端（官方协议 + 放宽 keepalive，见 exp40/policy_client.py）。"""
    from exp40.policy_client import Pi05PolicyClient

    return Pi05PolicyClient(host, port)


def model_input_summary(payload: dict) -> dict:
    return {
        'keys': sorted(k for k in payload if k != dc.KEY_PROMPT),
        'prompt': payload[dc.KEY_PROMPT],
        'shapes': {k: list(np.asarray(v).shape) for k, v in payload.items() if k != dc.KEY_PROMPT},
        'dtypes': {k: str(np.asarray(v).dtype) for k, v in payload.items() if k != dc.KEY_PROMPT},
        'sha256': {
            k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
            for k, v in payload.items() if k != dc.KEY_PROMPT and isinstance(v, np.ndarray)
        },
    }
