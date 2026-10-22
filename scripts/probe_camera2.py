"""为实验 2 的第三人称相机试几个视角，保存 PNG 供人工挑选。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np  # noqa: E402
import pybullet as p  # noqa: E402
from PIL import Image  # noqa: E402

from vla.atomic_tasks.scene2 import build_scene  # noqa: E402

CANDS = {
    'a_topdown': dict(target=(0.55, 0.19, 0.02), distance=0.95, yaw=-90, pitch=-89, fov=55),
    'b_p70': dict(target=(0.55, 0.19, 0.03), distance=1.00, yaw=-90, pitch=-70, fov=50),
    'c_p55': dict(target=(0.55, 0.19, 0.03), distance=1.00, yaw=-90, pitch=-55, fov=50),
    'd_yaw0': dict(target=(0.55, 0.19, 0.03), distance=1.00, yaw=0, pitch=-70, fov=50),
}


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else '/tmp/cam2')
    out.mkdir(parents=True, exist_ok=True)
    sim, block, wall, tray, info = build_scene()
    try:
        for name, cfg in CANDS.items():
            view = p.computeViewMatrixFromYawPitchRoll(
                list(cfg['target']), cfg['distance'], cfg['yaw'], cfg['pitch'], 0, 2)
            proj = p.computeProjectionMatrixFOV(cfg['fov'], 1.0, 0.02, 5.0)
            pixels = p.getCameraImage(512, 512, view, proj, renderer=p.ER_TINY_RENDERER,
                                      flags=p.ER_NO_SEGMENTATION_MASK,
                                      physicsClientId=sim.client)[2]
            rgb = np.asarray(pixels, dtype=np.uint8).reshape(512, 512, 4)[:, :, :3]
            Image.fromarray(rgb).save(out / f'{name}.png')
            print('saved', out / f'{name}.png')
    finally:
        p.disconnect(sim.client)


if __name__ == '__main__':
    main()
