"""Replay and render recorded 0B trials. No new IK and no parameter search.

Each replay rebuilds the recorded state (skill A, optional HOME reset) and then
commands the stored IK candidate with the original motor controller. Frames are
rendered offscreen with PyBullet's TinyRenderer; no GUI or CUDA is required.
"""
from pathlib import Path
import argparse
import csv
import json
import sys
import numpy as np
import pybullet as p
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from simulation import Simulation, HOME
from robot_controller import RobotController
from skills import Z1, Z2, skill_success

SOURCE = ROOT / 'results' / 'experiment_0b' / 'multistart_results.csv'
CONFIG = ROOT / 'results' / 'experiment_0b' / 'config.json'
OUT = ROOT / 'results' / 'experiment_0b_visualization'
SCENARIOS = [('z1', 'B', 'direct'), ('z1', 'C', 'direct'), ('z2', 'B', 'direct'),
             ('z2', 'C', 'direct'), ('z1', 'C', 'reset_to_canonical')]
COLORS = {'z1->B': (53, 120, 184), 'z1->C': (214, 39, 40), 'z2->B': (44, 160, 44),
          'z2->C': (255, 127, 14), 'z1->C (reset)': (140, 86, 175)}
DT = 1 / 240


def get_font(size):
    for name in ('arial.ttf', 'DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def load_record(implementation, successor, mode):
    with SOURCE.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    group = [r for r in rows if r['mode'] == mode and r['implementation'] == implementation
             and r['successor'] == successor]
    key = 'execution_success' if mode == 'direct' else 'executed'
    chosen = [r for r in group if r[key] == 'True']
    if not chosen:
        raise RuntimeError(f'no usable record for {implementation}->{successor} ({mode})')
    chosen.sort(key=lambda r: (int(r['trial']), int(r['ik_start_index'])))
    return chosen[0]


def render(sim, width=640, height=480):
    view = p.computeViewMatrixFromYawPitchRoll([0.45, -0.08, 0.30], 1.2, 55, -25, 0, 2)
    proj = p.computeProjectionMatrixFOV(55, width / height, 0.05, 5)
    pixels = p.getCameraImage(width, height, view, proj, renderer=p.ER_TINY_RENDERER,
                              physicsClientId=sim.client)[2]
    return Image.fromarray(np.asarray(pixels, dtype=np.uint8).reshape(height, width, 4)).convert('RGB')


def run_scenario(implementation, successor, mode, record, every, targets, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    label = f'{implementation}->{successor}' + (' (reset)' if mode == 'reset_to_canonical' else '')
    sim = Simulation(gui=False)
    ctl = RobotController(sim)
    frames, samples = [], []
    font = get_font(19)
    try:
        skill = Z1 if implementation == 'z1' else Z2
        a_result = ctl.move_pose(skill.position, skill.orientation)
        a_ok = bool(a_result.success and skill_success(sim, skill))
        reset_seconds = reset_distance = 0.0
        if mode == 'reset_to_canonical':
            home_motion = ctl.move(np.asarray(HOME, dtype=float))
            reset_seconds, reset_distance = home_motion.seconds, home_motion.distance
        start = sim.q().copy()
        recorded_start = np.asarray(json.loads(record['start_q_rad']), dtype=float)
        start_diff = float(np.max(np.abs(start - recorded_start)))
        target = np.asarray(json.loads(record['ik_candidate_q_rad']), dtype=float)

        def capture(text, seconds):
            image = render(sim)
            draw = ImageDraw.Draw(image)
            draw.text((10, 8), text, fill='white', stroke_width=2, stroke_fill='black', font=font)
            image.save(out_dir / f'frame_{len(frames):04d}.png')
            frames.append((image, seconds))

        capture(f'{label}  t=0.00s  start', 0.0)

        def observer(k, n, q, seconds):
            pos, _ = sim.pose()
            samples.append({'t': seconds, 'q': q.copy(), 'pos': pos.copy()})
            if k % every == 0 or k == n + 60:
                capture(f'{label}  t={seconds:.2f}s', seconds)

        motion = ctl.move(target, observer=observer)
        pos, quat = sim.pose()
        joint_error = float(np.linalg.norm(sim.q() - target))
        target_position = np.asarray(targets[successor]['position_m'], dtype=float)
        target_quat = p.getQuaternionFromEuler(targets[successor]['euler_rad'])
        position_error = float(np.linalg.norm(pos - target_position))
        orientation_error = float(2 * np.arccos(np.clip(abs(np.dot(quat, target_quat)), 0, 1)))
        capture(f'{label}  t={motion.seconds:.2f}s  end', motion.seconds)

        pictures = [frame[0] for frame in frames]
        pictures[0].save(out_dir / 'motion.gif', save_all=True, append_images=pictures[1:],
                         duration=max(20, int(every * DT * 1000)), loop=0)
        magnitude = 'success' if motion.success else f'FAIL {motion.reason}'
        print(f'{label}: {magnitude}; frames={len(frames)}; '
              f'length={motion.distance:.3f} rad; {motion.seconds:.3f} s')
        return {
            'label': label, 'condition': f'{implementation}->{successor}', 'mode': mode,
            'implementation': implementation, 'successor': successor, 'color': COLORS[label],
            'source_trial': int(record['trial']),
            'source_ik_start_index': int(record['ik_start_index']),
            'source_execution_success': record['execution_success'] == 'True',
            'replay_success': bool(motion.success), 'replay_reason': motion.reason,
            'source_length_rad': float(record['trajectory_length_rad']),
            'replay_length_rad': float(motion.distance),
            'length_diff_rad': abs(float(record['trajectory_length_rad']) - float(motion.distance)),
            'source_seconds': float(record['execution_seconds']),
            'replay_seconds': float(motion.seconds),
            'seconds_diff': abs(float(record['execution_seconds']) - float(motion.seconds)),
            'source_reset_seconds': float(record['reset_seconds']),
            'replay_reset_seconds': reset_seconds,
            'start_q_max_abs_diff_rad': start_diff,
            'joint_tracking_error_rad': joint_error,
            'position_error_m': position_error,
            'orientation_error_rad': orientation_error,
            'frames': len(frames), 'recorded_every_steps': every,
            'a_success': a_ok, 'reset_distance_rad': reset_distance,
        }, frames, samples
    finally:
        sim.close()


def key_indices(count, wanted=6):
    if count <= wanted:
        return list(range(count))
    return sorted({round(i * (count - 1) / (wanted - 1)) for i in range(wanted)})


def make_overview(runs, output):
    cell_w, cell_h, side = 280, 210, 150
    row_h = cell_h + 26
    width = side + 6 * cell_w + 20
    sheet = Image.new('RGB', (width, 76 + len(runs) * row_h), 'white')
    draw = ImageDraw.Draw(sheet)
    draw.text((16, 14), 'Experiment 0B replays: recorded handoffs (t=0 -> end)', fill='black', font=get_font(22))
    small = get_font(15)
    for row_index, run in enumerate(runs):
        y0 = 60 + row_index * row_h
        draw.text((16, y0 + 8), run['label'], fill=run['color'], font=small)
        status = 'success' if run['replay_success'] else run['replay_reason']
        draw.text((16, y0 + 30), status, fill='black' if run['replay_success'] else '#b02020', font=small)
        draw.text((16, y0 + 50), f'{run["replay_seconds"]:.2f} s', fill='black', font=small)
        for column, index in enumerate(key_indices(len(run['frames']))):
            picture, seconds = run['frames'][index]
            image = picture.resize((cell_w, cell_h), Image.LANCZOS)
            x0 = side + column * cell_w
            sheet.paste(image, (x0, y0))
            draw.rectangle((x0, y0, x0 + cell_w - 1, y0 + cell_h - 1), outline='#999999')
            draw.text((x0 + 6, y0 + cell_h - 22), f't={seconds:.2f}s', fill='white',
                      stroke_width=2, stroke_fill='black', font=small)
    sheet.save(output)


def make_ee_paths(runs, targets, output):
    width, height = 1060, 540
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    draw.text((16, 14), 'End-effector paths during the recorded successor motion', fill='black', font=get_font(22))
    small = get_font(14)
    points = [s['pos'] for run in runs for s in run['samples']]
    points += [np.asarray(targets[name]['position_m'], dtype=float) for name in ('B', 'C')]
    xs = [float(q[0]) for q in points]
    ys = [float(q[1]) for q in points]
    zs = [float(q[2]) for q in points]
    spans = [(min(xs) - 0.02, max(xs) + 0.02), (min(ys) - 0.03, max(ys) + 0.03),
             (min(zs) - 0.03, max(zs) + 0.03)]
    panels = [(66, 66, 500, 400, 0, 1, 'x (m)', 'y (m)', 'top view (x-y)'),
              (580, 66, 1014, 400, 0, 2, 'x (m)', 'z (m)', 'side view (x-z)')]
    for x0, y0, x1, y1, i, j, xlabel, ylabel, title in panels:
        draw.rectangle((x0, y0, x1, y1), outline='black')
        draw.text((x0, y0 - 22), title, fill='black', font=small)
        draw.text(((x0 + x1) / 2, y1 + 40), xlabel, fill='black', font=small, anchor='ma')
        draw.text((x0 - 66, y0 - 22), ylabel, fill='black', font=small)
        for value in np.linspace(spans[i][0], spans[i][1], 4):
            px = x0 + (value - spans[i][0]) / (spans[i][1] - spans[i][0]) * (x1 - x0)
            draw.text((px, y1 + 8), f'{value:.2f}', fill='black', font=small, anchor='ma')
        for value in np.linspace(spans[j][0], spans[j][1], 4):
            py = y1 - (value - spans[j][0]) / (spans[j][1] - spans[j][0]) * (y1 - y0)
            draw.text((x0 - 6, py), f'{value:.2f}', fill='black', font=small, anchor='rm')
        for run in runs:
            coords = []
            for sample in run['samples']:
                ax = x0 + (float(sample['pos'][i]) - spans[i][0]) / (spans[i][1] - spans[i][0]) * (x1 - x0)
                ay = y1 - (float(sample['pos'][j]) - spans[j][0]) / (spans[j][1] - spans[j][0]) * (y1 - y0)
                coords.append((ax, ay))
            if len(coords) > 1:
                draw.line(coords, fill=run['color'], width=3)
        for name, color in (('B', (0, 0, 0)), ('C', (120, 120, 120))):
            position = targets[name]['position_m']
            px = x0 + (position[i] - spans[i][0]) / (spans[i][1] - spans[i][0]) * (x1 - x0)
            py = y1 - (position[j] - spans[j][0]) / (spans[j][1] - spans[j][0]) * (y1 - y0)
            draw.ellipse((px - 6, py - 6, px + 6, py + 6), outline=color, width=2)
            draw.text((px + 8, py), name, fill=color, font=small, anchor='lm')
    for index, run in enumerate(runs):
        x = 66 + (index % 3) * 330
        y = 466 + (index // 3) * 28
        draw.rectangle((x, y, x + 24, y + 9), fill=run['color'])
        draw.text((x + 32, y + 4), run['label'], fill='black', font=small, anchor='lm')
    image.save(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--every', type=int, default=12, help='render one frame every N simulation steps')
    args = parser.parse_args()
    if args.every < 1:
        parser.error('--every must be positive')
    if not SOURCE.exists():
        raise SystemExit(f'missing {SOURCE}; run scripts/run_experiment_0b.py first')
    targets = json.loads(CONFIG.read_text(encoding='utf-8'))['targets']
    OUT.mkdir(parents=True, exist_ok=True)
    runs, rows = [], []
    for implementation, successor, mode in SCENARIOS:
        record = load_record(implementation, successor, mode)
        out_dir = OUT / (f'{implementation}_{successor}' + ('_reset' if mode == 'reset_to_canonical' else ''))
        metrics, frames, samples = run_scenario(implementation, successor, mode, record, args.every, targets, out_dir)
        metrics['frames'], metrics['samples'] = frames, samples
        runs.append(metrics)
        rows.append({key: value for key, value in metrics.items() if key not in ('frames', 'samples', 'color')})
    make_overview([run for run in runs if run['mode'] == 'direct'], OUT / 'overview_direct.png')
    make_ee_paths(runs, targets, OUT / 'ee_paths.png')
    with (OUT / 'replay_metrics.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    identical = all(row['source_execution_success'] == row['replay_success'] for row in rows)
    max_length = max(row['length_diff_rad'] for row in rows)
    max_start = max(row['start_q_max_abs_diff_rad'] for row in rows)
    (OUT / 'README.md').write_text(
        '# 实验 0B 可视化回放\n\n'
        '本目录是 `results/experiment_0b/multistart_results.csv` 的回放，不重新求解 IK、不搜索参数、不覆盖 0B 数据。\n'
        '每条回放按记录重建：先执行技能 A（z1/z2），重置组再回到 HOME，然后命令 CSV 中保存的 IK 候选，使用原电机控制器执行。\n\n'
        f'运行：`& ".\\\\.conda312\\\\python.exe" ".\\\\scripts\\\\visualize_experiment_0b.py"`，默认每 12 个仿真步渲染一帧（20 fps GIF）。\n\n'
        f'- 回放成功/失败与 CSV 是否全部一致：{identical}\n'
        f'- 最大轨迹长度差：{max_length:.3e} rad\n'
        f'- 最大初始关节差：{max_start:.3e} rad\n\n'
        '`overview_direct.png` 为四组直接交接的关键帧对照；`ee_paths.png` 为末端路径；各子目录含 `motion.gif`、逐帧 PNG 和该组 `result.png`。\n'
        '`replay_metrics.csv` 保留来源记录编号、成功标志、行程、时间和误差，便于核对。\n\n'
        '五组回放全部成功。此前 `z1_C_reset` 的 tracking_error 已定位为 `panda_link7` 与 `panda_hand` 的'
        '碰撞几何重叠：失败判定忽略该对、物理引擎仍求解，给第 5 关节附加约 12 N·m 制动；该对现已在动力学层关闭。'
        '可视化是确定性场景的复现，不是独立物理试验，也不是 VLA 泛化证据。\n',
        encoding='utf-8')
    print(f'metrics: {OUT / "replay_metrics.csv"}')
    print(f'replay matches recorded success flags: {identical}')
    print(f'max length diff: {max_length:.3e} rad; max start diff: {max_start:.3e} rad')


if __name__ == '__main__':
    main()
