"""临时脚本：搜索能同时覆盖起点、放置点与手臂起始位姿的第三人称相机参数。"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pybullet as p

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from grasp_task import DEFAULT_PARAMS
from manipulation import create_block, enable_contact_solver
from robot_controller import RobotController
from simulation import Simulation

sim = Simulation(gui=False, reference_cube=False)
enable_contact_solver(sim)
create_block(sim, DEFAULT_PARAMS['block_half'], DEFAULT_PARAMS['block_mass'],
             DEFAULT_PARAMS['block_start'])
ee_home = np.round(sim.pose()[0], 3)
ctl = RobotController(sim)
points = {
    'start': np.array([0.55, 0.0, 0.04]),
    'start_j': np.array([0.56, 0.01, 0.09]),
    'place': np.array([0.45, 0.18, 0.04]),
    'place_high': np.array([0.45, 0.18, 0.20]),
    'home_ee': ee_home,
    'home_arm': np.array([0.30, 0.0, 0.30]),
    'retreat': np.array([0.45, 0.18, 0.24]),
}
print('home ee', ee_home)


def proj(view, proj_matrix, point):
    clip = np.asarray(proj_matrix).reshape(4, 4).T @ np.asarray(view).reshape(4, 4).T @ np.array([*point, 1.0])
    ndc = clip[:3] / clip[3]
    return (ndc[0] * 0.5 + 0.5) * 256, (0.5 - ndc[1] * 0.5) * 256


best = []
for distance, fov, pitch, tz, yaw in itertools.product(
        (0.75, 0.85, 0.95, 1.05), (34, 38, 42, 46, 50), (-30, -38, -46, -54, -62, -70),
        (0.03, 0.06, 0.09, 0.12), (35, 45, 55, 65, 75)):
    view = p.computeViewMatrixFromYawPitchRoll([0.48, 0.08, tz], distance, yaw, pitch, 0, 2)
    proj_matrix = p.computeProjectionMatrixFOV(fov, 1.0, 0.02, 5.0)
    coords = {k: proj(view, proj_matrix, v) for k, v in points.items()
              if k not in ('home_ee', 'home_arm')}
    inside = all(12 <= u <= 244 and 12 <= v <= 244 for u, v in coords.values())
    if not inside:
        continue
    corners = [proj(view, proj_matrix, (0.55 + sx * 0.02, 0.0 + sy * 0.02, 0.081))
               for sx in (-1, 1) for sy in (-1, 1)]
    block_px = max(np.hypot(a[0] - b[0], a[1] - b[1]) for a in corners for b in corners)
    margin = min(min(u, 256 - u, v, 256 - v) for u, v in coords.values())
    best.append((fov, distance, pitch, tz, yaw, round(block_px, 1), round(margin, 1), coords))
best.sort(key=lambda item: (-item[5], item[0]))
for row in best[:8]:
    print(row[:7])
if best:
    print('coords of preferred:', {k: tuple(np.round(v, 0)) for k, v in best[0][7].items()})
p.disconnect(sim.client)
