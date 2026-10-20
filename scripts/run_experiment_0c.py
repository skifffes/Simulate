"""实验 0C 第一阶段：真实接触抓取-抬升-移动-放置。

单场景，重复执行以检查可重复性；首次执行保存关键帧、GIF、物体轨迹与
接触记录。所有动作由关节电机与夹爪电机驱动，工件只通过接触被搬运。

用法:
    python scripts/run_experiment_0c.py               # 5 次重复 + 可视化
    python scripts/run_experiment_0c.py --episodes 1  # 只跑一次
    python scripts/run_experiment_0c.py --no-render   # 只出数据
"""
from pathlib import Path
import argparse
import csv
import json
import sys

import numpy as np
import pybullet as p
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from simulation import Simulation
from robot_controller import RobotController
from manipulation import create_block, enable_contact_solver
from grasp_task import DEFAULT_PARAMS, GraspPlaceTask

OUT = ROOT / 'results' / 'experiment_0c'
CAMERA = {'target': (0.47, 0.07, 0.13), 'distance': 1.0, 'yaw': 55, 'pitch': -28}
FRAME_SIZE = (640, 480)
GIF_SIZE = (480, 360)
FRAME_EVERY = 12  # 240 Hz / 12 = 20 fps 回放


def render(sim, width=960, height=720):
    view = p.computeViewMatrixFromYawPitchRoll(
        list(CAMERA['target']), CAMERA['distance'], CAMERA['yaw'], CAMERA['pitch'], 0, 2)
    proj = p.computeProjectionMatrixFOV(52, width / height, 0.02, 5)
    pixels = p.getCameraImage(width, height, view, proj, renderer=p.ER_TINY_RENDERER,
                              physicsClientId=sim.client)[2]
    return Image.fromarray(np.asarray(pixels, dtype=np.uint8).reshape(height, width, 4)).convert('RGB')


def build_scene(params):
    sim = Simulation(gui=False, reference_cube=False)
    enable_contact_solver(sim)
    yaw = float(params.get('block_yaw', 0.0))
    block = create_block(sim, params['block_half'], params['block_mass'],
                         params['block_start'], orientation=p.getQuaternionFromEuler([0, 0, yaw]),
                         lateral_friction=params['block_friction'])
    sim.allowed_contacts = {(block, 9), (block, 10)}
    return sim, block


def make_params(index, jitter):
    """jitter>0 时随机平移/转动工件起点，并让抓取点跟随工件（假定已定位）。"""
    params = {key: (tuple(value) if isinstance(value, tuple) else value)
              for key, value in DEFAULT_PARAMS.items()}
    params['block_yaw'] = 0.0
    if jitter <= 0:
        return params, 0.0
    rng = np.random.default_rng(20261020 + index)
    dx, dy = rng.uniform(-jitter, jitter, 2)
    yaw = float(rng.uniform(-0.10, 0.10))
    base = DEFAULT_PARAMS['block_start']
    params['block_start'] = (base[0] + dx, base[1] + dy, base[2])
    params['block_yaw'] = yaw
    for key in ('pre_grasp', 'grasp', 'lift'):
        z = DEFAULT_PARAMS[key][2]
        params[key] = (params['block_start'][0], params['block_start'][1], z)
    return params, float(np.hypot(dx, dy))


def apply_overrides(params, args):
    """命令行覆盖摩擦与夹持力，用于复现失败样例。"""
    if getattr(args, 'friction', None) is not None:
        params['block_friction'] = float(args.friction)
    if getattr(args, 'force', None) is not None:
        params['grasp_force'] = float(args.force)
    return params


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_trajectory(path, samples):
    write_csv(path, samples, ['t', 'stage', 'ee_x', 'ee_y', 'ee_z',
                              'obj_x', 'obj_y', 'obj_z', 'obj_lin_speed', 'obj_ang_speed',
                              'tilt_deg', 'finger1', 'finger2',
                              'contact_force_1', 'contact_force_2', 'contact_links'])


def save_contacts(path, contacts):
    write_csv(path, contacts, ['t', 'stage', 'link', 'normal_force', 'px', 'py', 'pz'])


def save_keyframe_grid(images, meta, path, columns=4, thumb=(480, 360)):
    if not images:
        return
    rows = int(np.ceil(len(images) / columns))
    cell_w, cell_h = thumb
    canvas = Image.new('RGB', (cell_w * columns, cell_h * rows), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (image, (stage, seconds)) in enumerate(zip(images, meta)):
        cell = image.resize(thumb, Image.LANCZOS)
        x, y = (i % columns) * cell_w, (i // columns) * cell_h
        canvas.paste(cell, (x, y))
        draw.rectangle([x, y, x + cell_w - 1, y + cell_h - 1], outline=(40, 40, 40))
        draw.rectangle([x, y, x + 240, y + 24], fill=(0, 0, 0))
        draw.text((x + 8, y + 6), f'{stage}  t={seconds:.2f}s', fill=(255, 255, 255))
    canvas.save(path)


def run_episode(index, capture=None, params=None):
    """capture=(frames_dir, key_dir) 时保存画面；否则只算物理。"""
    params = params or DEFAULT_PARAMS
    sim, block = build_scene(params)
    gif_frames, key_images, key_meta = [], [], []
    state = {'stage': None, 'count': 0}

    def on_step(stage, k, n):
        if capture is None:
            return
        frames_dir, key_dir = capture
        if stage != state['stage']:
            state['stage'] = stage
            image = render(sim)
            key_images.append(image)
            key_meta.append((stage, sim.steps * sim.dt))
            if key_dir is not None:
                image.resize((640, 480), Image.LANCZOS).save(key_dir / f'{len(key_images):02d}_{stage}.png')
        if sim.steps % FRAME_EVERY == 0:
            image = render(sim, *FRAME_SIZE)
            state['count'] += 1
            if frames_dir is not None:
                image.save(frames_dir / f'{state["count"]:04d}.png')
            gif_frames.append(image.resize(GIF_SIZE, Image.LANCZOS))

    task = GraspPlaceTask(sim, RobotController(sim), block, params=params, on_step=on_step)
    task.run()
    result = task.result()
    if capture is not None and not result['success']:
        # 失败发生在阶段检查处，补一帧“失败瞬间”，否则关键帧停在最后一次阶段切换。
        image = render(sim)
        draw = ImageDraw.Draw(image)
        draw.rectangle([0, 0, image.width, 34], fill=(140, 20, 20))
        draw.text((8, 9), f"FAILED: {result['failure']}", fill=(255, 255, 255))
        key_images.append(image)
        key_meta.append(('failure', sim.steps * sim.dt))
        if capture[1] is not None:
            image.resize((640, 480), Image.LANCZOS).save(
                capture[1] / f'{len(key_images):02d}_failure.png')
        gif_frames.append(image.resize(GIF_SIZE, Image.LANCZOS))
    result['episode'] = index
    result['stage_trace'] = task.stages
    result['samples'] = task.samples
    result['contacts'] = task.contacts
    result['gif_frames'] = gif_frames
    result['key_images'] = key_images
    result['key_meta'] = key_meta
    sim.close()
    return result


def summarise(results):
    rows = []
    for r in results:
        rows.append({
            'episode': r['episode'],
            'block_offset_m': r['block_offset_m'],
            'success': r['success'],
            'failure': r['failure'] or '',
            'lift_height_m': r['lift_height_m'],
            'slip_m': r['slip_m'],
            'place_xy_error_m': r['place_xy_error_m'],
            'place_height_error_m': r['place_height_error_m'],
            'final_tilt_deg': r['final_tilt_deg'],
            'final_linear_speed': r['final_linear_speed'],
            'final_angular_speed': r['final_angular_speed'],
            'release_contacts': r['release_contacts'],
            'closed_finger_1': r['closed_finger_widths'][0],
            'closed_finger_2': r['closed_finger_widths'][1],
        })
    return rows


def stat(rows, key):
    values = [float(r[key]) for r in rows if np.isfinite(float(r[key]))]
    if not values:
        return None  # 例如全部在抬升阶段失败，从未到达放置点
    return {'mean': float(np.mean(values)), 'std': float(np.std(values)),
            'min': float(np.min(values)), 'max': float(np.max(values))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=5, help='固定场景重复次数')
    parser.add_argument('--no-render', action='store_true', help='不保存 GIF/关键帧')
    parser.add_argument('--jitter', type=float, default=0.0,
                        help='工件起点随机平移幅度（米），0 表示固定场景')
    parser.add_argument('--friction', type=float, default=None,
                        help='工件摩擦系数（默认 1.2；给 0.2 可复现摩擦不足的失败）')
    parser.add_argument('--force', type=float, default=None,
                        help='每指夹持力上限 N（默认 20；给 2 可复现夹持不足的失败）')
    parser.add_argument('--out', default=str(OUT), help='输出目录')
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frames_dir, key_dir = out / 'frames', out / 'keyframes'
    if not args.no_render:
        for directory in (frames_dir, key_dir):
            directory.mkdir(exist_ok=True)
            for old in directory.glob('*.png'):
                old.unlink()

    results = []
    for index in range(args.episodes):
        capture = None if args.no_render or index > 0 else (frames_dir, key_dir)
        params, offset = make_params(index, args.jitter)
        params = apply_overrides(params, args)
        result = run_episode(index, capture, params)
        result['block_offset_m'] = offset
        results.append(result)
        place = result['place_xy_error_m']
        place_txt = f'{place * 1000:.1f} mm' if np.isfinite(place) else '未到达'
        print(f"[{index + 1}/{args.episodes}] offset={offset * 1000:.1f} mm "
              f"success={result['success']} failure={result['failure']} "
              f"lift={result['lift_height_m']:.3f} m "
              f"place_err={place_txt} "
              f"speed={result['final_linear_speed']:.4f}")

    first = results[0]
    rows = summarise(results)
    save_trajectory(out / 'trajectory.csv', first['samples'])
    save_contacts(out / 'contacts.csv', first['contacts'])
    write_csv(out / 'repeatability.csv', rows, list(rows[0].keys()))
    payload = {k: v for k, v in first.items()
               if k not in ('samples', 'contacts', 'gif_frames', 'key_images', 'key_meta')}
    (out / 'result.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')

    if first['gif_frames']:
        first['gif_frames'][0].save(out / 'grasp.gif', save_all=True,
                                    append_images=first['gif_frames'][1:],
                                    duration=50, loop=0, optimize=True)
        save_keyframe_grid(first['key_images'], first['key_meta'], out / 'keyframes.png')

    success_rate = float(np.mean([r['success'] for r in rows]))
    summary = {
        'episodes': args.episodes,
        'jitter_m': args.jitter,
        'success_rate': success_rate,
        'failures': sorted({r['failure'] for r in rows if r['failure']}),
        'lift_height_m': stat(rows, 'lift_height_m'),
        'slip_m': stat(rows, 'slip_m'),
        'place_xy_error_m': stat(rows, 'place_xy_error_m'),
        'final_tilt_deg': stat(rows, 'final_tilt_deg'),
        'final_linear_speed': stat(rows, 'final_linear_speed'),
        'params': first['params'],
        'note': '同一固定场景的确定性重复运行，用于检查可重复性，不是独立物理样本。',
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'success rate {success_rate * 100:.0f}%  ->  {out}')


if __name__ == '__main__':
    main()
