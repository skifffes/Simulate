"""观测适配器：PyBullet 观测 → π₀.₅ DROID 策略输入。

原则（对应 prompt 的 Step 3）：
1. 只做官方客户端明确做过的变换（resize_with_pad 到 224×224、RGB uint8、原样传关节角）。
2. 不做“为了让维度对上”的填充；8 维 state 的每一位都有物理含义（7 关节 + 1 夹爪）。
3. 每个字段都返回一份 sha256 与统计，供审计与复现。
4. 相机语义差异必须显式记录：Simulate 的 top 是 45°/30° 斜视第三人称相机，
   DROID 的 exterior_image_1_left 是左外侧相机。二者不是同一个视角，
   实验 4.0 只做接口连通，不声称视觉分布一致。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

try:  # 官方客户端的实现；仿真侧未安装时用本文件的等价实现
    from openpi_client import image_tools as _image_tools
except ImportError:  # pragma: no cover
    _image_tools = None

from . import droid_contract as dc

FINGER_MAX_M = 0.04  # panda_finger_joint1/2 的上限（m），simulation.py 中 URDF 限位


def resize_with_pad(images: np.ndarray, height: int, width: int) -> np.ndarray:
    """与 openpi_client.image_tools.resize_with_pad 等价（PIL BILINEAR + 零填充）。"""
    if _image_tools is not None:
        return _image_tools.resize_with_pad(images, height, width)

    from PIL import Image

    if images.shape[-3:-1] == (height, width):
        return images
    original = images.shape
    flat = images.reshape(-1, *original[-3:])
    out = []
    for im in flat:
        img = Image.fromarray(im)
        cur_w, cur_h = img.size
        ratio = max(cur_w / width, cur_h / height)
        new_w, new_h = int(cur_w / ratio), int(cur_h / ratio)
        resized = img.resize((new_w, new_h), resample=Image.BILINEAR)
        canvas = Image.new(resized.mode, (width, height), 0)
        canvas.paste(resized, (max(0, int((width - new_w) / 2)), max(0, int((height - new_h) / 2))))
        out.append(np.asarray(canvas))
    resized_arr = np.stack(out)
    return resized_arr.reshape(*original[:-3], *resized_arr.shape[-3:])


def finger_widths(sim) -> np.ndarray:
    """两指当前位移（m）。与 vla 阶段 `finger_widths` 的定义一致：单指相对 0 的位移。"""
    import pybullet as p

    return np.array([p.getJointState(sim.robot, 9, physicsClientId=sim.client)[0],
                     p.getJointState(sim.robot, 10, physicsClientId=sim.client)[0]], dtype=np.float64)


def gripper_scalar(sim) -> float:
    """把两指状态压成 π₀.₅ 需要的 1 维夹爪量。

    DROID 的 gripper_position ∈ [0, 1] 是归一化开合量（0=闭合，1=全开）。
    PyBullet 单指位移范围是 [0, 0.04] m，两指同目标、总开口 = 2×单指位移，
    因此用 单指位移 / 0.04 线性映射，并夹到 [1e-4, 0.9999] 以避免 0/1 端点。
    """
    width = float(np.mean(finger_widths(sim)))
    g = width / FINGER_MAX_M
    return float(np.clip(g, 1e-4, 0.9999))


def _sha256(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(arr).tobytes())
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    return h.hexdigest()


@dataclass
class ObservationPacket:
    """一次策略查询的完整输入记录（原始 + 实际发送）。"""

    payload: dict
    audit: dict = field(default_factory=dict)

    def keys(self) -> list[str]:
        return sorted(self.payload)


def build_droid_observation(
    frames: dict,
    q: np.ndarray,
    gripper: float,
    prompt: str,
    *,
    resize_to: tuple[int, int] = dc.CLIENT_IMAGE_HW,
    keep_raw: bool = True,
) -> ObservationPacket:
    """组装 DROID 客户端请求体。

    frames: {'top': (H,W,3) uint8, 'wrist': (H,W,3) uint8}，来自 vla.cameras.CameraRig.capture()
    q: (7,) 关节角（rad）
    gripper: [0,1] 归一化夹爪量
    """
    for name in ('top', 'wrist'):
        if name not in frames:
            raise KeyError(f'缺少相机 {name}；当前只有 {sorted(frames)}')
    q = np.asarray(q, dtype=np.float32).reshape(-1)
    if q.shape != (7,):
        raise ValueError(f'关节角必须是 7 维，收到 {q.shape}')
    if not np.all(np.isfinite(q)):
        raise ValueError('关节角包含 NaN/Inf')

    images = {}
    image_stats = {}
    for key, name in ((dc.KEY_IMAGE_BASE, 'top'), (dc.KEY_IMAGE_WRIST, 'wrist')):
        img = np.asarray(frames[name])
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f'{name} 图像必须是 (H,W,3)，收到 {img.shape}')
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        sent = resize_with_pad(img, resize_to[0], resize_to[1])
        if sent.dtype != np.uint8:
            raise ValueError('resize 后 dtype 必须是 uint8')
        images[key] = sent
        image_stats[key] = {
            'camera': name,
            'raw_shape': list(np.asarray(frames[name]).shape),
            'sent_shape': list(sent.shape),
            'dtype': str(sent.dtype),
            'min': int(sent.min()), 'max': int(sent.max()),
            'mean': float(sent.mean()), 'std': float(sent.std()),
            'sha256_sent': _sha256(sent),
        }
        if keep_raw:
            image_stats[key]['sha256_raw'] = _sha256(np.asarray(frames[name]).astype(np.uint8))

    payload = {
        dc.KEY_IMAGE_BASE: images[dc.KEY_IMAGE_BASE],
        dc.KEY_IMAGE_WRIST: images[dc.KEY_IMAGE_WRIST],
        dc.KEY_JOINT_POSITION: q,
        dc.KEY_GRIPPER_POSITION: np.array([gripper], dtype=np.float32),
        dc.KEY_PROMPT: str(prompt),
    }
    audit = {
        'state_dim': int(1 + q.size),
        'state_order': ['panda_joint1..7 (rad)', 'gripper_normalized[0,1]'],
        'state': q.tolist() + [float(gripper)],
        'gripper_raw_finger_m': float(gripper) * FINGER_MAX_M,
        'prompt': str(prompt),
        'images': image_stats,
        'keys': sorted(k for k in payload if k != dc.KEY_PROMPT),
    }
    return ObservationPacket(payload=payload, audit=audit)
