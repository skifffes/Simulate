"""临时脚本：对比腕部相机候选参数在接近/抓取/抬起阶段的成像。"""
import sys
from pathlib import Path

import numpy as np
import pybullet as p
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from grasp_task import DEFAULT_PARAMS
from manipulation import create_block, enable_contact_solver, set_gripper
from robot_controller import RobotController
from simulation import Simulation
from vla.cameras import CameraRig

OUT = ROOT / 'results/experiment_1/diagnostics/camera_probe'
OUT.mkdir(parents=True, exist_ok=True)

TOP = {'type': 'fixed', 'target': (0.48, 0.08, 0.09), 'distance': 0.75,
       'yaw': 45.0, 'pitch': -30.0, 'fov': 34.0, 'near': 0.02, 'far': 5.0}

CANDIDATES = {
    'A_f78_down04': {'offset': (0.06, 0.0, -0.045), 'aim': 'grasp', 'look_ahead': 0.02,
                     'target_down': 0.04, 'fov': 78.0},
    'B_f75_down03': {'offset': (0.07, 0.0, -0.02), 'aim': 'grasp', 'look_ahead': 0.02,
                     'target_down': 0.03, 'fov': 75.0},
    'C_f75_down02': {'offset': (0.09, 0.0, -0.01), 'aim': 'grasp', 'look_ahead': 0.02,
                     'target_down': 0.02, 'fov': 75.0},
    'D_f82_down05': {'offset': (0.06, 0.0, -0.06), 'aim': 'grasp', 'look_ahead': 0.02,
                     'target_down': 0.05, 'fov': 82.0},
    'E_f70_down05_high': {'offset': (0.05, 0.0, -0.02), 'aim': 'grasp', 'look_ahead': 0.03,
                          'target_down': 0.05, 'fov': 72.0},
}


def wrist_cfg(spec):
    cfg = {'type': 'wrist', 'link': 8, 'offset': spec['offset'], 'fov': spec['fov'],
           'near': 0.005, 'far': 3.0, 'aim': spec['aim'],
           'look_ahead': spec['look_ahead'], 'target_down': spec['target_down']}
    return cfg


def snapshot(rig, tag):
    frames = rig.capture()
    for name, frame in frames.items():
        Image.fromarray(frame).save(OUT / f'{tag}_{name}.png')
    return frames['wrist']


sim = Simulation(gui=False, reference_cube=False)
enable_contact_solver(sim)
block = create_block(sim, DEFAULT_PARAMS['block_half'], DEFAULT_PARAMS['block_mass'],
                     DEFAULT_PARAMS['block_start'])
sim.allowed_contacts = {(block, 9), (block, 10)}
ctl = RobotController(sim)

rigs = {}
for name, spec in CANDIDATES.items():
    rigs[name] = CameraRig(sim, {'top': TOP, 'wrist': wrist_cfg(spec)}, (256, 256), supersample=1)

shots = {name: [] for name in rigs}
tags = []


def capture_all(tag):
    tags.append(tag)
    for name, rig in rigs.items():
        shots[name].append(snapshot(rig, f'{tag}_{name}'))


capture_all('home')
set_gripper(sim, 0.04, 20, 0.06)
ctl.move_pose(DEFAULT_PARAMS['pre_grasp'], DEFAULT_PARAMS['orientation'])
capture_all('pregrasp')
set_gripper(sim, 0.024, 20, 0.05)
ctl.move_pose(DEFAULT_PARAMS['grasp'], DEFAULT_PARAMS['orientation'])
capture_all('grasp')
set_gripper(sim, 0.017, 20, 0.05)
for _ in range(int(0.8 / sim.dt)):
    sim.step()
capture_all('closed')
ctl.move_pose(DEFAULT_PARAMS['lift'], DEFAULT_PARAMS['orientation'])
capture_all('lift')

tile, margin, header = 160, 6, 18
for name, frames in shots.items():
    canvas = Image.new('RGB', (len(frames) * (tile + margin) + margin,
                               tile + margin * 2 + header), (18, 18, 20))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 4), f'{name}  {CANDIDATES[name]}', fill=(235, 235, 235))
    for column, (frame, tag) in enumerate(zip(frames, tags)):
        left = margin + column * (tile + margin)
        canvas.paste(Image.fromarray(frame).resize((tile, tile), Image.BILINEAR), (left, header + margin))
        draw.text((left + 2, header + margin + tile - 14), tag, fill=(255, 220, 120))
    canvas.save(OUT / f'grid_{name}.png')

print('done')
p.disconnect(sim.client)
