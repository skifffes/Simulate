#!/usr/bin/env python
"""实验 0D：计划条件化的物理技能交接验证（主实验）。

阶段一 A：z1 / z2 两种真实抓放终态（长边分别沿 X / Y）；
阶段二 B：闭合夹爪侧面推动到挡板；C：顶抓提升并保持；
组合臂：z1->B、z2->B、z1->C、z2->C；
基线臂：固定 z1、固定 z2（由组合臂聚合）、恢复公共标准状态后执行后继。

所有成功率、时间与关节行程都来自仿真执行；失败案例全部保留并写入 CSV。
"""
import argparse
import csv
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pybullet as p
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from manipulation import (  # noqa: E402
    create_block, enable_contact_solver, finger_contact_forces, object_state,
)
from reset_primitives import joint_travel, reset_to_canonical  # noqa: E402
from robot_controller import RobotController  # noqa: E402
from simulation import Simulation  # noqa: E402
from successor_tasks import PushTask, TopGraspTask, create_wall  # noqa: E402
from two_state_task import (  # noqa: E402
    DEFAULT_PARAMS, HALF, HANDOVER, HandoverTask, object_yaw_deg,
)

OUT = ROOT / 'results' / 'experiment_0d'
CAMERA = dict(target=(0.48, 0.12, 0.10), distance=1.15, yaw=60, pitch=-30,
              width=960, height=720)

HANDOVER_YAW = {'z1': 0.0, 'z2': 90.0}
ARMS = {
    'z1_B': {'handover': 'z1', 'successor': 'B', 'reset': False},
    'z2_B': {'handover': 'z2', 'successor': 'B', 'reset': False},
    'z1_C': {'handover': 'z1', 'successor': 'C', 'reset': False},
    'z2_C': {'handover': 'z2', 'successor': 'C', 'reset': False},
    'reset_B': {'handover': 'z2', 'successor': 'B', 'reset': True},
    'reset_C': {'handover': 'z2', 'successor': 'C', 'reset': True},
}
ARM_ORDER = list(ARMS)

PERTURBATION = {
    'start_x_m': (0.547, 0.553),
    'start_y_m': (-0.003, 0.003),
    'start_yaw_deg': (-3.0, 3.0),
    'mass_kg': (0.075, 0.085),
    'lateral_friction': (1.05, 1.35),
}

FONT_CANDIDATES = [
    'C:/Windows/Fonts/msyh.ttc',
    'C:/Windows/Fonts/simhei.ttf',
    'C:/Windows/Fonts/simsun.ttc',
]


def font(size):
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def sample_config(seed):
    """每个 episode 的扰动：所有实验臂共用同一配置，构成配对比较。"""
    rng = np.random.default_rng(int(seed))
    return {
        'start': [
            float(rng.uniform(*PERTURBATION['start_x_m'])),
            float(rng.uniform(*PERTURBATION['start_y_m'])),
            0.021,
        ],
        'yaw_deg': float(rng.uniform(*PERTURBATION['start_yaw_deg'])),
        'mass': float(rng.uniform(*PERTURBATION['mass_kg'])),
        'friction': float(rng.uniform(*PERTURBATION['lateral_friction'])),
    }


def build_scene(cfg, gui=False):
    """所有臂使用完全相同的环境：工件、挡板、接触过滤。"""
    sim = Simulation(gui=gui, realtime=False, reference_cube=False)
    enable_contact_solver(sim)
    orientation = p.getQuaternionFromEuler([0.0, 0.0, math.radians(cfg['yaw_deg'])])
    block = create_block(sim, HALF, cfg['mass'], cfg['start'],
                         orientation=orientation,
                         lateral_friction=cfg['friction'],
                         color=(0.85, 0.35, 0.10, 1.0))
    sim.allowed_contacts = {(block, 9), (block, 10)}
    wall = create_wall(sim)
    controller = RobotController(sim)
    return sim, controller, block, wall


class FrameGrabber:
    """按时间间隔抓帧，并在阶段切换时保存关键帧。"""

    def __init__(self, sim, root, every=120):
        self.sim = sim
        self.root = Path(root)
        self.frames = self.root / 'frames'
        self.keys = self.root / 'keyframes'
        self.frames.mkdir(parents=True, exist_ok=True)
        self.keys.mkdir(parents=True, exist_ok=True)
        self.every = int(every)
        self.last_stage = None
        self.key_index = 0

    def __call__(self, stage, k, n):
        if self.sim.steps % self.every == 0:
            self.sim.screenshot(self.frames / f'{self.sim.steps:06d}.png', **CAMERA)
        if stage != self.last_stage:
            self.last_stage = stage
            self.sim.screenshot(self.keys / f'{self.key_index:02d}_{stage}.png', **CAMERA)
            self.key_index += 1

    def make_gif(self, path, duration=120):
        files = sorted(self.frames.glob('*.png'))
        if not files:
            return None
        images = [Image.open(f).convert('RGB').resize((640, 480)) for f in files]
        path = Path(path)
        images[0].save(path, save_all=True, append_images=images[1:],
                       duration=duration, loop=0, optimize=True)
        return path


def handover_snapshot(sim, block, target_yaw_deg):
    obj = object_state(sim, block)
    yaw = float(object_yaw_deg(sim, block))
    diff = (yaw - target_yaw_deg + 180.0) % 360.0 - 180.0
    f1, f2, active = finger_contact_forces(sim, block)
    return {
        'x': float(obj['position'][0]),
        'y': float(obj['position'][1]),
        'z': float(obj['position'][2]),
        'yaw_deg': yaw,
        'yaw_error_deg': float(diff),
        'tilt_deg': float(obj['tilt_deg']),
        'finger_force_1': float(f1),
        'finger_force_2': float(f2),
        'finger_contact_links': len(active),
    }


def run_episode(arm_key, episode, seed, cfg, frames_root=None, write_files=True):
    arm = ARMS[arm_key]
    sim, ctl, block, wall = build_scene(cfg)
    grabber = FrameGrabber(sim, frames_root) if frames_root else None
    wall_clock = time.perf_counter()
    t0 = sim.steps
    try:
        a_task = HandoverTask(sim, ctl, block, HANDOVER_YAW[arm['handover']],
                              on_step=grabber)
        a_ok = a_task.run()
        a_res = a_task.result()
        handover = handover_snapshot(sim, block, HANDOVER_YAW[arm['handover']])
        a_joint = joint_travel(a_res)

        reset_ok = None
        reset_res = None
        if arm['reset'] and a_ok:
            reset_ok, reset_res = reset_to_canonical(sim, ctl, block, on_step=grabber)
            handover['post_reset_yaw_deg'] = float(object_yaw_deg(sim, block))
            handover['post_reset_tilt_deg'] = float(
                object_state(sim, block)['tilt_deg'])

        succ_entered = bool(a_ok and (reset_ok is None or reset_ok))
        succ_ok = False
        succ_res = {}
        succ_contacts = []
        if succ_entered:
            if arm['successor'] == 'B':
                succ_task = PushTask(sim, ctl, block, on_step=grabber)
                succ_task.wall = wall
            else:
                succ_task = TopGraspTask(sim, ctl, block, on_step=grabber)
            succ_ok = succ_task.run()
            succ_res = succ_task.result()
            succ_contacts = succ_task.contacts

        collisions = len(sim.collisions())
        a_s = float(a_res['sim_seconds'])
        r_s = float(reset_res['sim_seconds']) if reset_res else 0.0
        s_s = float(succ_res.get('sim_seconds', 0.0))
        r_j = joint_travel(reset_res) if reset_res else 0.0
        s_j = joint_travel(succ_res) if succ_res else 0.0

        if not a_ok:
            failure = f"A:{a_res['failure_text']} ({a_res['failure_stage']})"
        elif reset_ok is False:
            failure = f"reset:{reset_res['failure_text']} ({reset_res['failure_stage']})"
        elif succ_entered and not succ_ok:
            failure = (f"{arm['successor']}:{succ_res.get('failure_text')} "
                       f"({succ_res.get('failure_stage')})")
        else:
            failure = ''

        row = {
            'arm': arm_key,
            'handover': arm['handover'],
            'successor': arm['successor'],
            'uses_reset': int(bool(arm['reset'])),
            'episode': episode,
            'seed': seed,
            'start_x_m': cfg['start'][0],
            'start_y_m': cfg['start'][1],
            'start_yaw_deg': cfg['yaw_deg'],
            'mass_kg': cfg['mass'],
            'friction': cfg['friction'],
            'a_success': int(a_ok),
            'a_failure': a_res['failure'] or '',
            'a_failure_stage': a_res['failure_stage'] or '',
            'a_failure_text': a_res['failure_text'] or '',
            'a_sim_s': a_s,
            'a_joint_rad': a_joint,
            'handover_x_m': handover['x'],
            'handover_y_m': handover['y'],
            'handover_z_m': handover['z'],
            'handover_yaw_deg': handover['yaw_deg'],
            'handover_yaw_error_deg': handover['yaw_error_deg'],
            'handover_tilt_deg': handover['tilt_deg'],
            'handover_force_1_n': handover['finger_force_1'],
            'handover_force_2_n': handover['finger_force_2'],
            'handover_contact_links': handover['finger_contact_links'],
            'handover_closed_widths': str(a_res['closed_finger_widths']),
            'reset_called': int(bool(arm['reset'] and a_ok)),
            'reset_success': '' if reset_ok is None else int(reset_ok),
            'reset_failure': '' if not reset_res else (reset_res['failure'] or ''),
            'reset_failure_text': '' if not reset_res else (reset_res['failure_text'] or ''),
            'reset_sim_s': r_s,
            'reset_joint_rad': r_j,
            'post_reset_yaw_deg': handover.get('post_reset_yaw_deg', ''),
            'succ_entered': int(succ_entered),
            'succ_success': int(succ_ok),
            'succ_failure': succ_res.get('failure') or '',
            'succ_failure_stage': succ_res.get('failure_stage') or '',
            'succ_failure_text': succ_res.get('failure_text') or '',
            'succ_sim_s': s_s,
            'succ_joint_rad': s_j,
            'displacement_m': succ_res.get('displacement_m', ''),
            'wall_gap_m': succ_res.get('wall_gap_m', ''),
            'face_contact_samples': succ_res.get('face_contact_samples', ''),
            'lift_rise_m': succ_res.get('lift_rise_m', ''),
            'slip_m': succ_res.get('slip_m', ''),
            'succ_final_tilt_deg': succ_res.get('final_tilt_deg', ''),
            'total_success': int(bool(a_ok and succ_ok
                                      and (reset_ok is None or reset_ok))),
            'total_failure': failure,
            'total_sim_s': a_s + r_s + s_s,
            'total_joint_rad': a_joint + r_j + s_j,
            'collisions': collisions,
            'wall_clock_s': time.perf_counter() - wall_clock,
            'sim_steps': sim.steps - t0,
        }
        detail = {
            'row': row,
            'handover': handover,
            'stage_a': a_res,
            'stage_reset': reset_res,
            'stage_successor': succ_res,
            'a_contacts': a_task.contacts,
            'reset_contacts': (reset_res or {}).get('contacts', []),
            'succ_contacts': succ_contacts,
        }
    finally:
        sim.close()

    if write_files:
        detail_dir = OUT / 'details'
        contact_dir = OUT / 'contacts'
        detail_dir.mkdir(parents=True, exist_ok=True)
        contact_dir.mkdir(parents=True, exist_ok=True)
        with (detail_dir / f'{arm_key}_ep{episode}.json').open(
                'w', encoding='utf-8') as fh:
            json.dump(jsonable(detail), fh, ensure_ascii=False, indent=2)
        write_contacts(contact_dir / f'{arm_key}_ep{episode}.csv',
                       detail['a_contacts'], detail['reset_contacts'],
                       detail['succ_contacts'])
    return row, detail, grabber


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def write_contacts(path, a_contacts, reset_contacts, succ_contacts):
    fields = ['phase', 't', 'stage', 'link', 'normal_force', 'px', 'py', 'pz',
              'lx', 'ly', 'lz']
    with Path(path).open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for phase, contacts in (('A', a_contacts), ('reset', reset_contacts),
                                ('successor', succ_contacts)):
            for c in contacts:
                writer.writerow({k: c.get(k, '') for k in fields} | {'phase': phase})


CSV_FIELDS = [
    'arm', 'handover', 'successor', 'uses_reset', 'episode', 'seed',
    'start_x_m', 'start_y_m', 'start_yaw_deg', 'mass_kg', 'friction',
    'a_success', 'a_failure', 'a_failure_stage', 'a_failure_text',
    'a_sim_s', 'a_joint_rad',
    'handover_x_m', 'handover_y_m', 'handover_z_m', 'handover_yaw_deg',
    'handover_yaw_error_deg', 'handover_tilt_deg', 'handover_force_1_n',
    'handover_force_2_n', 'handover_contact_links', 'handover_closed_widths',
    'reset_called', 'reset_success', 'reset_failure', 'reset_failure_text',
    'reset_sim_s', 'reset_joint_rad', 'post_reset_yaw_deg',
    'succ_entered', 'succ_success', 'succ_failure', 'succ_failure_stage',
    'succ_failure_text', 'succ_sim_s', 'succ_joint_rad',
    'displacement_m', 'wall_gap_m', 'face_contact_samples', 'lift_rise_m',
    'slip_m', 'succ_final_tilt_deg',
    'total_success', 'total_failure', 'total_sim_s', 'total_joint_rad',
    'collisions', 'wall_clock_s', 'sim_steps',
]


def write_runs(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in CSV_FIELDS})


def summarize(rows):
    def mean(values):
        values = [float(v) for v in values if v != '']
        return float(np.mean(values)) if values else None

    def rate(values):
        values = [float(v) for v in values]
        return float(np.mean(values)) if values else None

    per_arm = {}
    for arm in ARM_ORDER:
        arm_rows = [r for r in rows if r['arm'] == arm]
        if not arm_rows:
            continue
        entered = [r for r in arm_rows if r['succ_entered']]
        per_arm[arm] = {
            'episodes': len(arm_rows),
            'a_success_rate': rate([r['a_success'] for r in arm_rows]),
            'reset_success_rate': rate([r['reset_success'] for r in arm_rows
                                        if r['reset_called']]),
            'successor_success_rate_given_entered': rate(
                [r['succ_success'] for r in entered]) if entered else None,
            'total_success_rate': rate([r['total_success'] for r in arm_rows]),
            'mean_a_sim_s': mean([r['a_sim_s'] for r in arm_rows]),
            'mean_reset_sim_s': mean([r['reset_sim_s'] for r in arm_rows]),
            'mean_succ_sim_s': mean([r['succ_sim_s'] for r in arm_rows]),
            'mean_total_sim_s': mean([r['total_sim_s'] for r in arm_rows]),
            'mean_total_joint_rad': mean([r['total_joint_rad'] for r in arm_rows]),
            'failures': [r['total_failure'] for r in arm_rows if not r['total_success']],
        }

    def combine(arms):
        sub = [r for r in rows if r['arm'] in arms]
        return {
            'arms': arms,
            'episodes': len(sub),
            'total_success_rate': rate([r['total_success'] for r in sub]),
            'mean_total_sim_s': mean([r['total_sim_s'] for r in sub]),
            'mean_total_joint_rad': mean([r['total_joint_rad'] for r in sub]),
        }

    return {
        'episodes_per_arm': len(rows) // len(ARM_ORDER) if rows else 0,
        'arms': per_arm,
        'baselines': {
            'baseline_1_fixed_z1': combine(['z1_B', 'z1_C']),
            'baseline_2_fixed_z2': combine(['z2_B', 'z2_C']),
            'baseline_3_canonical_restore': combine(['reset_B', 'reset_C']),
        },
    }


COLOR_A = (66, 120, 190)
COLOR_RESET = (232, 168, 66)
COLOR_SUCC = (74, 158, 108)
COLOR_FAIL = (198, 82, 72)
COLOR_GRID = (226, 229, 234)
COLOR_TEXT = (38, 43, 50)
COLOR_MUTED = (96, 103, 112)


def draw_panel(draw, box, title, labels, series, top, percent=False, ylabel=''):
    """series: [(名称, [值...], 颜色)]，按顺序自下而上堆叠。"""
    x0, y0, x1, y1 = box
    draw.rectangle([x0, y0, x1, y1], outline=(205, 209, 216), width=2)
    draw.text((x0, y0 - 30), title, font=font(24), fill=COLOR_TEXT)
    left, right = x0 + 78, x1 - 24
    top_px, bottom_px = y0 + 34, y1 - 78
    for g in range(5):
        y = bottom_px - (bottom_px - top_px) * g / 4
        draw.line([left, y, right, y], fill=COLOR_GRID)
        draw.text((left - 10, y), f'{top * g / 4:.2f}', font=font(15),
                  fill=COLOR_MUTED, anchor='rm')
    draw.text((x0 + 10, y0 + 8), ylabel, font=font(15), fill=COLOR_MUTED)
    slot = (right - left) / max(1, len(labels))
    totals = [sum(s[1][i] for s in series) for i in range(len(labels))]
    for i, label in enumerate(labels):
        x = left + slot * i + slot * 0.20
        w = slot * 0.60
        base = bottom_px
        for _, values, color in series:
            h = (values[i] / top) * (bottom_px - top_px)
            if h > 0.2:
                draw.rectangle([x, base - h, x + w, base], fill=color,
                               outline=(255, 255, 255))
                base -= h
        text = f'{totals[i] * 100:.0f}%' if percent else f'{totals[i]:.2f}'
        draw.text((x + w / 2, base - 8), text, font=font(17),
                  fill=COLOR_TEXT, anchor='mb')
        draw.text((x + w / 2, bottom_px + 10), label, font=font(16),
                  fill=COLOR_MUTED, anchor='ma')
    lx = left
    for name, _, color in series:
        draw.rectangle([lx, bottom_px + 48, lx + 16, bottom_px + 62], fill=color)
        draw.text((lx + 22, bottom_px + 55), name, font=font(15),
                  fill=COLOR_MUTED, anchor='lm')
        lx += 34 + draw.textlength(name, font=font(15))


def plot_arms(rows, summary, path):
    width, height = 1200, 1180
    image = Image.new('RGB', (width, height), (247, 248, 250))
    draw = ImageDraw.Draw(image)
    draw.text((40, 22), '实验 0D：各规划臂的成功率与执行代价（所有失败均保留）',
              font=font(28), fill=COLOR_TEXT)
    arms = [a for a in ARM_ORDER if a in summary['arms']]
    n = {a: summary['arms'][a]['episodes'] for a in arms}

    def counts(arm, predicate):
        sub = [r for r in rows if r['arm'] == arm]
        return sum(1 for r in sub if predicate(r)) / max(1, len(sub))

    success_series = [
        ('全链路成功', [counts(a, lambda r: r['total_success']) for a in arms], COLOR_SUCC),
        ('A 成功/后继失败', [counts(a, lambda r: r['a_success'] and not r['total_success'])
                             for a in arms], COLOR_RESET),
        ('A 本身失败', [counts(a, lambda r: not r['a_success']) for a in arms], COLOR_FAIL),
    ]
    time_series = [
        ('A 交接', [summary['arms'][a]['mean_a_sim_s'] for a in arms], COLOR_A),
        ('B3 复位', [summary['arms'][a]['mean_reset_sim_s'] for a in arms], COLOR_RESET),
        ('后继', [summary['arms'][a]['mean_succ_sim_s'] for a in arms], COLOR_SUCC),
    ]
    # 关节行程按阶段拆分：从每次运行的 stage_trace 累计
    joint_a, joint_r, joint_s = [], [], []
    for a in arms:
        sub = [r for r in rows if r['arm'] == a]
        joint_a.append(float(np.mean([r['a_joint_rad'] for r in sub])))
        joint_r.append(float(np.mean([r['reset_joint_rad'] for r in sub])))
        joint_s.append(float(np.mean([r['succ_joint_rad'] for r in sub])))
    joint_series = [
        ('A 交接', joint_a, COLOR_A),
        ('B3 复位', joint_r, COLOR_RESET),
        ('后继', joint_s, COLOR_SUCC),
    ]
    draw_panel(draw, (40, 90, width - 40, 450), '成功率', arms, success_series,
               1.0, percent=True, ylabel='比例')
    draw_panel(draw, (40, 520, width - 40, 850), '执行时间（仿真秒，均值）', arms,
               time_series, max(1.0, max(sum(s[1][i] for s in time_series)
                                          for i in range(len(arms))) * 1.2),
               ylabel='s')
    draw_panel(draw, (40, 920, width - 40, 1150), '关节行程（rad，均值）', arms,
               joint_series, max(1.0, max(sum(s[1][i] for s in joint_series)
                                           for i in range(len(arms))) * 1.2),
               ylabel='rad')
    image.save(path)


def plot_handover(rows, path):
    width, height = 1200, 640
    image = Image.new('RGB', (width, height), (247, 248, 250))
    draw = ImageDraw.Draw(image)
    draw.text((40, 22), '交接状态与后继失败原因', font=font(28), fill=COLOR_TEXT)

    # 左：交接后工件朝向散点
    box = (40, 110, 560, 590)
    draw.rectangle(box, outline=(205, 209, 216), width=2)
    draw.text((box[0], box[1] - 30), '交接后工件朝向（yaw）', font=font(22),
              fill=COLOR_TEXT)
    left, right, top_px, bottom_px = box[0] + 60, box[2] - 20, box[1] + 30, box[3] - 70
    lo, hi = -15.0, 105.0
    for v in (0, 30, 60, 90):
        y = bottom_px - (v - lo) / (hi - lo) * (bottom_px - top_px)
        draw.line([left, y, right, y], fill=COLOR_GRID)
        draw.text((left - 8, y), f'{v}', font=font(14), fill=COLOR_MUTED, anchor='rm')
    arms = [a for a in ARM_ORDER if any(r['arm'] == a for r in rows)]
    slot = (right - left) / max(1, len(arms))
    for i, arm in enumerate(arms):
        sub = [r for r in rows if r['arm'] == arm]
        cx = left + slot * (i + 0.5)
        for j, r in enumerate(sub):
            yaw = float(r['handover_yaw_deg'])
            y = bottom_px - (yaw - lo) / (hi - lo) * (bottom_px - top_px)
            jitter = ((j % 5) - 2) * 5
            color = COLOR_SUCC if r['total_success'] else COLOR_FAIL
            draw.ellipse([cx + jitter - 6, y - 6, cx + jitter + 6, y + 6],
                         fill=color)
        draw.text((cx, bottom_px + 12), arm, font=font(15), fill=COLOR_MUTED,
                  anchor='ma')
    draw.text((left, box[1] + 6), '绿=全链路成功，红=失败', font=font(15),
              fill=COLOR_MUTED)

    # 右：失败阶段计数
    box = (620, 110, width - 40, 590)
    draw.rectangle(box, outline=(205, 209, 216), width=2)
    draw.text((box[0], box[1] - 30), '失败发生在哪一阶段', font=font(22),
              fill=COLOR_TEXT)
    left, right, top_px, bottom_px = box[0] + 60, box[2] - 20, box[1] + 34, box[3] - 70
    phases = [
        ('A 失败', COLOR_FAIL, lambda r: not r['a_success']),
        ('B3 复位失败', COLOR_RESET, lambda r: r['reset_success'] == 0),
        ('后继失败', COLOR_A, lambda r: r['succ_entered'] and not r['succ_success']),
    ]
    values = {name: [] for name, _, _ in phases}
    for arm in arms:
        sub = [r for r in rows if r['arm'] == arm]
        for name, _, predicate in phases:
            values[name].append(sum(1 for r in sub if predicate(r)))
    top = max(1, max(sum(values[name][i] for name, _, _ in phases)
                     for i in range(len(arms))))
    for g in range(5):
        y = bottom_px - (bottom_px - top_px) * g / 4
        draw.line([left, y, right, y], fill=COLOR_GRID)
        draw.text((left - 8, y), f'{top * g / 4:.0f}', font=font(14),
                  fill=COLOR_MUTED, anchor='rm')
    slot = (right - left) / max(1, len(arms))
    for i, arm in enumerate(arms):
        x = left + slot * i + slot * 0.22
        w = slot * 0.56
        base = bottom_px
        for name, color, _ in phases:
            count = values[name][i]
            h = count / top * (bottom_px - top_px)
            if h > 0.5:
                draw.rectangle([x, base - h, x + w, base], fill=color,
                               outline=(255, 255, 255))
                base -= h
        total = sum(values[name][i] for name, _, _ in phases)
        draw.text((x + w / 2, base - 8), str(total), font=font(16),
                  fill=COLOR_TEXT, anchor='mb')
        draw.text((x + w / 2, bottom_px + 12), arm, font=font(15),
                  fill=COLOR_MUTED, anchor='ma')
    lx = left
    for name, color, _ in phases:
        draw.rectangle([lx, bottom_px + 46, lx + 16, bottom_px + 60], fill=color)
        draw.text((lx + 22, bottom_px + 53), name, font=font(15),
                  fill=COLOR_MUTED, anchor='lm')
        lx += 34 + draw.textlength(name, font=font(15))
    image.save(path)


def make_representative_gifs(rows, out):
    """对每个臂重放一条成功和一条失败（若存在），生成 GIF 与关键帧。"""
    gif_dir = out / 'gifs'
    gif_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for arm in ARM_ORDER:
        sub = [r for r in rows if r['arm'] == arm]
        for outcome, label in ((1, 'success'), (0, 'failure')):
            pick = next((r for r in sub if r['total_success'] == outcome), None)
            if pick is None:
                continue
            cfg = {
                'start': [pick['start_x_m'], pick['start_y_m'], 0.021],
                'yaw_deg': pick['start_yaw_deg'],
                'mass': pick['mass_kg'],
                'friction': pick['friction'],
            }
            name = f'{arm}_{label}_ep{pick["episode"]}'
            work = gif_dir / f'_work_{name}'
            replay, _, grabber = run_episode(
                arm, pick['episode'], pick['seed'], cfg,
                frames_root=work, write_files=False)
            gif_path = grabber.make_gif(gif_dir / f'{name}.gif')
            key_dst = out / 'keyframes' / name
            if grabber.keys.exists():
                key_dst.mkdir(parents=True, exist_ok=True)
                for frame in grabber.keys.glob('*.png'):
                    shutil.move(str(frame), key_dst / frame.name)
            shutil.rmtree(work, ignore_errors=True)
            made.append({
                'name': name,
                'arm': arm,
                'outcome': label,
                'episode': int(pick['episode']),
                'recorded_success': int(pick['total_success']),
                'replay_success': int(replay['total_success']),
                'gif': str(gif_path),
                'keyframes': str(key_dst),
            })
            print(f'  GIF {name}: 复现{"成功" if replay["total_success"] else "失败"}')
    with (gif_dir / 'index.json').open('w', encoding='utf-8') as fh:
        json.dump(made, fh, ensure_ascii=False, indent=2)
    return made


def print_summary(summary, gifs):
    print('\n臂             A成功率  后继(进入后)  全链路  总时间(s)  关节(rad)  失败数')
    for arm in ARM_ORDER:
        data = summary['arms'].get(arm)
        if not data:
            continue
        succ = data['successor_success_rate_given_entered']
        print(f"{arm:<12} {data['a_success_rate']*100:6.0f}%  "
              f"{(succ if succ is not None else 0)*100:10.0f}%  "
              f"{data['total_success_rate']*100:5.0f}%  "
              f"{data['mean_total_sim_s']:8.2f}  "
              f"{data['mean_total_joint_rad']:9.2f}  "
              f"{len(data['failures']):4d}")
    print('\n基线（成功率 / 平均总时间 / 平均关节行程）:')
    for name, data in summary['baselines'].items():
        print(f"  {name:<32} {data['total_success_rate']*100:5.0f}%  "
              f"{data['mean_total_sim_s']:7.2f} s  "
              f"{data['mean_total_joint_rad']:8.2f} rad")
    print(f'\nGIF/关键帧: {len(gifs)} 组')


def main():
    global OUT
    parser = argparse.ArgumentParser(description='实验 0D 主实验')
    parser.add_argument('--episodes', type=int, default=5,
                        help='每个臂的 episode 数（扰动种子 = seed + episode）')
    parser.add_argument('--seed', type=int, default=20261021)
    parser.add_argument('--arms', default=','.join(ARM_ORDER),
                        help='逗号分隔的实验臂')
    parser.add_argument('--tag', default='', help='结果写入 results/experiment_0d/<tag>')
    parser.add_argument('--no-gifs', action='store_true', help='跳过 GIF 重放')
    args = parser.parse_args()

    arms = [a.strip() for a in args.arms.split(',') if a.strip()]
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise SystemExit(f'未知实验臂: {unknown}')

    out = OUT / args.tag if args.tag else OUT
    OUT = out
    for sub in ('details', 'contacts', 'gifs', 'keyframes'):
        (out / sub).mkdir(parents=True, exist_ok=True)

    config = {
        'experiment': '0D plan-conditioned physical skill handover',
        'seed_base': args.seed,
        'episodes_per_arm': args.episodes,
        'arms': ARMS,
        'handover_yaw_deg': HANDOVER_YAW,
        'block_half_extents_m': list(HALF),
        'handover_point_xy': list(HANDOVER),
        'task_params': jsonable(DEFAULT_PARAMS),
        'perturbation': PERTURBATION,
        'note': ('每 episode 的扰动配置由 seed+episode 生成，所有臂共用，'
                 '因此臂间是配对比较；失败案例不筛除。'),
    }
    with (out / 'config.json').open('w', encoding='utf-8') as fh:
        json.dump(config, fh, ensure_ascii=False, indent=2)

    rows = []
    total = args.episodes * len(arms)
    print(f'开始实验 0D：{len(arms)} 臂 x {args.episodes} episode = {total} 次')
    for episode in range(args.episodes):
        seed = args.seed + episode
        cfg = sample_config(seed)
        for arm in arms:
            row, _, _ = run_episode(arm, episode, seed, cfg)
            rows.append(row)
            write_runs(out / 'runs.csv', rows)
            status = '成功' if row['total_success'] else f"失败({row['total_failure']})"
            print(f"  ep{episode} {arm:<8} A={'Y' if row['a_success'] else 'N'} "
                  f"{status}  {row['total_sim_s']:.1f}s  "
                  f"yaw={row['handover_yaw_deg']:.1f}")

    summary = summarize(rows)
    with (out / 'summary.json').open('w', encoding='utf-8') as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    plot_arms(rows, summary, out / 'fig_0d_arms.png')
    plot_handover(rows, out / 'fig_0d_handover.png')
    gifs = [] if args.no_gifs else make_representative_gifs(rows, out)
    print_summary(summary, gifs)
    print(f'\n结果目录: {out}')


if __name__ == '__main__':
    main()
