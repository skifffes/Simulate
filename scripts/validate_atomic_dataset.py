"""校验 LeRobot v2.1 数据集，并做动作回放验证。

检查项：
1. 元数据：info.json 的版本/频率/路径模板/统计数与磁盘实际内容一致。
2. 逐 episode：时间戳步长、frame_index、episode_index、全局 index 连续性、
   task_index 与 tasks.jsonl 对应、parquet 列齐全。
3. 动作与状态：数值有限、关节限位、夹爪宽度范围、动作跳变。
4. 图文同步：动作是"下一周期观测"的指令，检查 |a_t - s_{t+1}| < |a_t - s_t|，
   并给出跟踪相关性；视频帧数与 parquet 长度一致、分辨率正确。
5. episodes_stats 与 parquet 实算统计一致。
6. 动作回放：DIRECT 模式下重放动作序列，比较重放关节状态与数据集状态。
7. 随机抽查：导出若干 episode 的抽帧拼图与动作/状态曲线。

用法：
    python scripts/validate_atomic_dataset.py --dataset data/lerobot_panda_pick_place
"""
import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

STATE_NAMES = [f'joint{i + 1}' for i in range(7)] + ['gripper']
ACTION_NAMES = STATE_NAMES
ARM_DIM = 7


# --------------------------------------------------------------------------- #
# 基础读取
# --------------------------------------------------------------------------- #
def read_jsonl(path):
    rows = []
    with Path(path).open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_meta(root):
    root = Path(root)
    info = json.loads((root / 'meta/info.json').read_text(encoding='utf-8'))
    return {
        'info': info,
        'episodes': read_jsonl(root / 'meta/episodes.jsonl'),
        'stats': read_jsonl(root / 'meta/episodes_stats.jsonl'),
        'tasks': read_jsonl(root / 'meta/tasks.jsonl'),
    }


def read_table(root, info, episode_index):
    import pyarrow.parquet as pq
    path = Path(root) / info['data_path'].format(
        episode_chunk=episode_index // info['chunks_size'], episode_index=episode_index)
    return pq.read_table(path)


def video_path(root, info, video_key, episode_index):
    return Path(root) / info['video_path'].format(
        episode_chunk=episode_index // info['chunks_size'],
        video_key=video_key, episode_index=episode_index)


def probe_video(path):
    """返回 (帧数, 时长秒, 宽, 高)。

    优先用 PyAV：服务器上 imageio_ffmpeg 的 count_frames_and_secs 会卡住
    （2026-10-26 实测），没有 av 时回退旧实现。
    """
    try:
        import av
    except ImportError:
        av = None
    if av is not None:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            frames = int(stream.frames or 0)
            if frames <= 0:
                frames = sum(1 for _ in container.decode(video=0))
            fps = float(stream.average_rate) if stream.average_rate else 0.0
            seconds = frames / fps if fps else 0.0
            return frames, float(seconds), int(stream.width), int(stream.height)
    import imageio_ffmpeg
    frames, seconds = imageio_ffmpeg.count_frames_and_secs(str(path))
    generator = imageio_ffmpeg.read_frames(str(path), pix_fmt='rgb24')
    meta = next(generator)
    width, height = meta['size']
    generator.close()
    return int(frames), float(seconds), int(width), int(height)


def read_video_frames(path, indices):
    """按需解码指定帧号（升序）。"""
    wanted = sorted(int(i) for i in indices)
    frames = {}
    try:
        import av
    except ImportError:
        av = None
    if av is not None:
        wanted_set = set(wanted)
        last = wanted[-1] if wanted else -1
        with av.open(str(path)) as container:
            for index, frame in enumerate(container.decode(video=0)):
                if index in wanted_set:
                    frames[index] = frame.to_ndarray(format='rgb24')
                if index >= last:
                    break
        return frames
    import imageio_ffmpeg
    cursor = 0
    generator = imageio_ffmpeg.read_frames(str(path), pix_fmt='rgb24')
    meta = next(generator)
    width, height = meta['size']
    try:
        for index, chunk in enumerate(generator):
            if cursor >= len(wanted):
                break
            if index == wanted[cursor]:
                frames[index] = np.frombuffer(chunk, dtype=np.uint8).reshape(height, width, 3)
                cursor += 1
    finally:
        generator.close()
    return frames


def joint_limits():
    import pybullet as p
    import pybullet_data
    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    body = p.loadURDF('franka_panda/panda.urdf', useFixedBase=True, physicsClientId=client)
    lower = np.array([p.getJointInfo(body, j, physicsClientId=client)[8] for j in range(ARM_DIM)])
    upper = np.array([p.getJointInfo(body, j, physicsClientId=client)[9] for j in range(ARM_DIM)])
    p.disconnect(client)
    return lower, upper


# --------------------------------------------------------------------------- #
# 检查结果收集
# --------------------------------------------------------------------------- #
class Report:
    def __init__(self):
        self.rows = []

    def check(self, scope, name, ok, detail=''):
        self.rows.append({'scope': scope, 'check': name, 'ok': bool(ok), 'detail': str(detail)})
        return bool(ok)

    def failures(self):
        return [row for row in self.rows if not row['ok']]

    def dump_csv(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=['scope', 'check', 'ok', 'detail'])
            writer.writeheader()
            writer.writerows(self.rows)


# --------------------------------------------------------------------------- #
# 元数据检查
# --------------------------------------------------------------------------- #
def check_meta(root, meta, report, expect_fps=None):
    info = meta['info']
    root = Path(root)
    episodes, stats, tasks = meta['episodes'], meta['stats'], meta['tasks']

    report.check('meta', 'codebase_version == v2.1', info.get('codebase_version') == 'v2.1',
                 info.get('codebase_version'))
    report.check('meta', 'chunks_size == 1000', info.get('chunks_size') == 1000, info.get('chunks_size'))
    if expect_fps is not None:
        report.check('meta', f'fps == {expect_fps}', info.get('fps') == expect_fps, info.get('fps'))
    report.check('meta', 'total_episodes 与 episodes.jsonl 一致',
                 info.get('total_episodes') == len(episodes),
                 f"{info.get('total_episodes')} vs {len(episodes)}")
    report.check('meta', 'total_tasks 与 tasks.jsonl 一致',
                 info.get('total_tasks') == len(tasks), f"{info.get('total_tasks')} vs {len(tasks)}")
    total_frames = sum(int(ep['length']) for ep in episodes)
    report.check('meta', 'total_frames 与各 episode 长度之和一致',
                 info.get('total_frames') == total_frames, f"{info.get('total_frames')} vs {total_frames}")

    video_keys = [key for key, spec in info['features'].items() if spec.get('dtype') == 'video']
    report.check('meta', '存在两路视频特征', len(video_keys) == 2, ','.join(video_keys))
    report.check('meta', 'total_videos 与 episode*视频数一致',
                 info.get('total_videos') == len(episodes) * len(video_keys),
                 f"{info.get('total_videos')} vs {len(episodes) * len(video_keys)}")
    expect_chunks = math.ceil(len(episodes) / info['chunks_size']) if episodes else 0
    report.check('meta', 'total_chunks 与 episode 数一致',
                 info.get('total_chunks') == expect_chunks, f"{info.get('total_chunks')} vs {expect_chunks}")
    report.check('meta', 'splits.train 覆盖全部 episode',
                 info.get('splits', {}).get('train') == f'0:{len(episodes)}', info.get('splits'))

    spec = info['features'].get('observation.state', {})
    report.check('meta', 'observation.state 为 7 关节 + 2 指（9 维）',
                 list(spec.get('shape', [])) == [ARM_DIM + 2], spec.get('shape'))
    spec = info['features'].get('action', {})
    report.check('meta', 'action 为 7 关节目标 + 夹爪目标（8 维）',
                 list(spec.get('shape', [])) == [ARM_DIM + 1], spec.get('shape'))
    for key in video_keys:
        spec = info['features'][key]
        report.check('meta', f'特征 {key} 为 256x256 RGB',
                     list(spec['shape']) == [256, 256, 3], spec['shape'])

    indices = [int(ep['episode_index']) for ep in episodes]
    report.check('meta', 'episode_index 连续无重复', indices == list(range(len(episodes))),
                 f'{indices[:3]}...{indices[-3:]}' if indices else 'empty')
    stat_indices = [int(row['episode_index']) for row in stats]
    report.check('meta', 'episodes_stats 与 episodes 一一对应', stat_indices == indices,
                 f'{len(stats)} vs {len(episodes)}')

    for row in stats:
        missing = [key for key in info['features'] if key not in row['stats']]
        if missing:
            report.check(f"episode {row['episode_index']}", 'episodes_stats 覆盖全部特征', False, missing)
            break
    else:
        report.check('meta', 'episodes_stats 覆盖全部特征', True)

    task_strings = sorted(task['task'] for task in tasks)
    report.check('meta', 'tasks 文本非空且唯一',
                 all(task.strip() for task in task_strings) and len(set(task_strings)) == len(task_strings),
                 task_strings[:1])
    return video_keys


# --------------------------------------------------------------------------- #
# 单 episode 检查
# --------------------------------------------------------------------------- #
def check_episode(root, meta, episode, stats_row, tasks, limits, video_keys, expect_fps,
                  report, check_video=True, index_cursor=None):
    info = meta['info']
    index = int(episode['episode_index'])
    scope = f'episode {index}'
    failures = []

    def fail(name, detail=''):
        failures.append(f'{name}:{detail}')
        return report.check(scope, name, False, detail)

    try:
        table = read_table(root, info, index)
    except Exception as error:  # noqa: BLE001
        fail('parquet 可读', error)
        return {'episode_index': index, 'failures': failures}

    columns = set(table.column_names)
    missing = [name for name in ('observation.state', 'action', 'timestamp', 'frame_index',
                                 'episode_index', 'index', 'task_index') if name not in columns]
    if missing:
        fail('parquet 列齐全', missing)
        return {'episode_index': index, 'failures': failures}

    states = np.asarray(table['observation.state'].to_pylist(), dtype=np.float64)
    actions = np.asarray(table['action'].to_pylist(), dtype=np.float64)
    timestamps = table['timestamp'].to_numpy(zero_copy_only=False).astype(np.float64)
    frame_index = table['frame_index'].to_numpy(zero_copy_only=False).astype(np.int64)
    ep_index = table['episode_index'].to_numpy(zero_copy_only=False).astype(np.int64)
    global_index = table['index'].to_numpy(zero_copy_only=False).astype(np.int64)
    task_index = table['task_index'].to_numpy(zero_copy_only=False).astype(np.int64)

    length = len(timestamps)
    report.check(scope, '长度与 episodes.jsonl 一致', length == int(episode['length']),
                 f"{length} vs {episode['length']}")
    state_dim = int(info['features']['observation.state']['shape'][0])
    action_dim = int(info['features']['action']['shape'][0])
    report.check(scope, '状态/动作形状与特征定义一致',
                 states.shape == (length, state_dim) and actions.shape == (length, action_dim),
                 f'{states.shape} {actions.shape} 期望 (T,{state_dim}) (T,{action_dim})')

    dt = 1.0 / expect_fps
    report.check(scope, '时间戳从 0 开始且步长 1/fps',
                 length > 0 and abs(timestamps[0]) < 1e-6
                 and np.allclose(np.diff(timestamps), dt, atol=1e-4),
                 f'diff={np.unique(np.round(np.diff(timestamps), 6))[:3] if length > 1 else []}')
    report.check(scope, 'frame_index 为 0..T-1', np.array_equal(frame_index, np.arange(length)),
                 f'{frame_index[:3]}...' if length else '')
    report.check(scope, 'episode_index 列恒定', np.all(ep_index == index), np.unique(ep_index))
    report.check(scope, 'index 列全局连续', np.array_equal(global_index, global_index[0] + np.arange(length)),
                 f'{global_index[0]}..{global_index[-1]}')
    report.check(scope, 'task_index 合法',
                 task_index.size > 0 and np.all(task_index == task_index[0])
                 and 0 <= int(task_index[0]) < max(1, len(tasks)),
                 np.unique(task_index))
    if index_cursor is not None:
        expected = index_cursor[0]
        report.check(scope, 'index 跨 episode 连续', int(global_index[0]) == expected,
                     f'{global_index[0]} vs {expected}')
        index_cursor[0] = int(global_index[-1]) + 1
    report.check(scope, 'episode 任务文本与 tasks 一致',
                 bool(episode.get('tasks')) and episode['tasks'][0] in {task['task'] for task in tasks},
                 episode.get('tasks'))

    finite = bool(np.isfinite(states).all() and np.isfinite(actions).all())
    report.check(scope, '状态/动作无 NaN/Inf', finite)
    if not finite:
        return {'episode_index': index, 'failures': failures + ['non_finite']}

    lower, upper = limits
    margin = 1e-3
    arm_ok = bool(np.all(actions[:, :ARM_DIM] >= lower - margin)
                  and np.all(actions[:, :ARM_DIM] <= upper + margin)
                  and np.all(states[:, :ARM_DIM] >= lower - margin)
                  and np.all(states[:, :ARM_DIM] <= upper + margin))
    report.check(scope, '机械臂关节在限位内', arm_ok,
                 f'violate={int(np.sum((actions[:, :ARM_DIM] < lower) | (actions[:, :ARM_DIM] > upper)))}')

    grip = np.concatenate([actions[:, ARM_DIM], states[:, ARM_DIM:].ravel()])
    report.check(scope, '夹爪宽度在 [0, 0.08]', bool(np.all((grip >= -1e-3) & (grip <= 0.08 + 1e-3))),
                 f'min={grip.min():.4f} max={grip.max():.4f}')

    jumps = np.abs(np.diff(actions[:, :ARM_DIM], axis=0)) if length > 1 else np.zeros((1, ARM_DIM))
    max_jump = float(jumps.max()) if jumps.size else 0.0
    report.check(scope, '相邻动作无异常跳变 (<0.35 rad/周期)', max_jump < 0.35, f'{max_jump:.3f}')

    # ---- 动作与下一观测的同步性 ----
    if length > 1:
        err_now = float(np.mean(np.abs(actions[:-1, :ARM_DIM] - states[:-1, :ARM_DIM])))
        err_next = float(np.mean(np.abs(actions[:-1, :ARM_DIM] - states[1:, :ARM_DIM])))
        report.check(scope, '动作指向下一周期观测', err_next <= err_now + 1e-3,
                     f'err_next={err_next:.4f} err_now={err_now:.4f}')
    else:
        err_now = err_next = 0.0

    moves = np.diff(states[:, :ARM_DIM], axis=0) if length > 1 else np.zeros((1, ARM_DIM))
    targets = actions[:-1, :ARM_DIM] - states[:-1, :ARM_DIM] if length > 1 else np.zeros((1, ARM_DIM))
    flat_moves, flat_targets = moves.ravel(), targets.ravel()
    if flat_moves.std() > 1e-6 and flat_targets.std() > 1e-6:
        corr = float(np.corrcoef(flat_moves, flat_targets)[0, 1])
    else:
        corr = 1.0
    report.check(scope, '运动方向与动作残差正相关 (r>0.5)', corr > 0.5, f'r={corr:.3f}')

    # ---- 统计一致性 ----
    if stats_row is None:
        report.check(scope, 'episodes_stats 存在', False)

    else:
        stats = stats_row['stats']
        ok_all, detail = True, ''
        for key, array in (('observation.state', states), ('action', actions)):
            entry = stats.get(key)
            if entry is None:
                ok_all, detail = False, f'{key} 缺失'
                break
            for name, reducer in (('min', np.min), ('max', np.max), ('mean', np.mean), ('std', np.std)):
                expected = reducer(array, axis=0)
                stored = np.asarray(entry[name], dtype=np.float64)
                if stored.shape != expected.shape or not np.allclose(stored, expected, atol=2e-3, rtol=1e-3):
                    ok_all, detail = False, f'{key}.{name}'
                    break
            if not ok_all:
                break
            count = np.asarray(entry['count']).ravel()
            if count.size != 1 or int(count[0]) != length:
                ok_all, detail = False, f'{key}.count={count}'
                break
        report.check(scope, 'episodes_stats 数值与实算一致', ok_all, detail)
        for key in video_keys:
            entry = stats.get(key, {})
            count = np.asarray(entry.get('count', [-1])).ravel()
            report.check(scope, f'视频统计 {key} 帧数一致', count.size == 1 and int(count[0]) == length,
                         count)

    # ---- 视频 ----
    video_info = {}
    for key in video_keys:
        path = video_path(root, info, key, index)
        if not path.exists():
            report.check(scope, f'视频存在 {key}', False, path.name)
            continue
        if not check_video:
            video_info[key] = {'path': str(path)}
            continue
        try:
            frames, seconds, width, height = probe_video(path)
        except Exception as error:  # noqa: BLE001
            report.check(scope, f'视频可解码 {key}', False, error)
            continue
        report.check(scope, f'视频帧数与长度一致 {key}', int(frames) == length, f'{frames} vs {length}')
        report.check(scope, f'视频分辨率 256x256 {key}', (width, height) == (256, 256), f'{width}x{height}')
        duration_ok = abs(float(seconds) - length / expect_fps) <= max(0.25, 2.0 / expect_fps)
        report.check(scope, f'视频时长与 fps 一致 {key}', duration_ok,
                     f'{seconds:.2f}s vs {length / expect_fps:.2f}s')
        video_info[key] = {'path': str(path), 'frames': int(frames), 'seconds': float(seconds)}

    return {
        'episode_index': index,
        'length': length,
        'err_now': err_now,
        'err_next': err_next,
        'corr': corr,
        'max_action_jump': max_jump,
        'states': states,
        'actions': actions,
        'videos': video_info,
        'failures': failures,
    }


# --------------------------------------------------------------------------- #
# 动作回放
# --------------------------------------------------------------------------- #
def replay_episode(states, actions, steps_per_action, settle_first=True):
    """在 DIRECT 场景中重放动作，返回与数据集状态的偏差。"""
    import pybullet as p

    from grasp_task import DEFAULT_PARAMS
    from manipulation import finger_widths, set_gripper
    from simulation import Simulation

    sim = Simulation(gui=False, reference_cube=False)
    errors_q, errors_grip = [], []
    try:
        for joint, value in enumerate(states[0][:ARM_DIM]):
            p.resetJointState(sim.robot, joint, float(value), 0.0, physicsClientId=sim.client)
        set_gripper(sim, float(actions[0][ARM_DIM]), DEFAULT_PARAMS['grasp_force'],
                    DEFAULT_PARAMS['close_speed'])
        sim.command(states[0][:ARM_DIM])
        if settle_first:
            for _ in range(steps_per_action):
                sim.step()
        for t in range(len(actions) - 1):
            sim.command(actions[t][:ARM_DIM])
            set_gripper(sim, float(actions[t][ARM_DIM]), DEFAULT_PARAMS['grasp_force'],
                        DEFAULT_PARAMS['close_speed'])
            for _ in range(steps_per_action):
                sim.step()
            errors_q.append(np.abs(sim.q() - states[t + 1][:ARM_DIM]))
            errors_grip.append(np.abs(np.asarray(finger_widths(sim)) - states[t + 1, ARM_DIM:]))
    finally:
        sim.close()

    errors_q = np.asarray(errors_q).reshape(-1, ARM_DIM)
    errors_grip = np.asarray(errors_grip).reshape(-1, 2)
    return {
        'mean_q_error': float(errors_q.mean()) if errors_q.size else 0.0,
        'max_q_error': float(errors_q.max()) if errors_q.size else 0.0,
        'mean_grip_error': float(errors_grip.mean()) if errors_grip.size else 0.0,
        'per_joint_mean': errors_q.mean(axis=0).round(5).tolist() if errors_q.size else [],
    }


# --------------------------------------------------------------------------- #
# 可视化抽查
# --------------------------------------------------------------------------- #
COLORS = [(255, 118, 117), (116, 192, 252), (255, 206, 84), (150, 206, 180),
          (206, 158, 255), (255, 159, 64), (144, 238, 144)]


def _panel(draw, box, series, colors, title, ylabel, legend_labels=None):
    """在 box=(left, top, right, bottom) 内画折线，纵轴按该面板整体范围缩放。"""
    left, top, right, bottom = box
    draw.text((left, top - 16), title, fill=(235, 235, 235))
    draw.rectangle([left, top, right, bottom], outline=(80, 80, 88))
    values = np.concatenate([np.asarray(s, dtype=float) for s in series]) if series else np.zeros(1)
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-6:
        low, high = low - 1e-3, high + 1e-3
    span = high - low
    low, high = low - span * 0.08, high + span * 0.08
    span = high - low
    for step in range(5):
        y = bottom - (bottom - top) * step / 4
        draw.line([(left, y), (right, y)], fill=(45, 45, 50))
    longest = max((len(s) for s in series), default=1)
    for values_1d, color in zip(series, colors):
        if len(values_1d) < 2:
            continue
        points = []
        for i, value in enumerate(values_1d):
            x = left + (right - left) * (i / (longest - 1))
            y = bottom - (bottom - top) * ((float(value) - low) / span)
            points.append((x, y))
        draw.line(points, fill=color, width=2)
    # 刻度最后画，避免被曲线覆盖
    for step in range(5):
        y = bottom - (bottom - top) * step / 4
        text = f'{low + span * step / 4:.2f}'
        width = draw.textlength(text)
        draw.rectangle([left - 10 - width, y - 6, left - 4, y + 6], fill=(20, 20, 23))
        draw.text((left - 10 - width, y - 6), text, fill=(140, 140, 150))
    draw.text((left, bottom + 4), ylabel, fill=(150, 150, 160))
    if legend_labels:
        x = right - 64
        panel = [x - 4, top + 2, right - 3, top + 4 + len(legend_labels) * 12]
        draw.rectangle(panel, fill=(20, 20, 23))
        for row, (label, color) in enumerate(zip(legend_labels, colors)):
            y = top + 4 + row * 12
            draw.line([(x, y + 4), (x + 12, y + 4)], fill=color, width=2)
            draw.text((x + 16, y), label, fill=(210, 210, 215))


def plot_episode(result, out_dir):
    from PIL import Image, ImageDraw

    out_dir.mkdir(parents=True, exist_ok=True)
    index = result['episode_index']
    states, actions = result['states'], result['actions']
    width, panel_height = 1100, 240
    left, right = 74, width - 16
    step = panel_height + 34

    canvas = Image.new('RGB', (width, 44 + 2 * step), (20, 20, 23))
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 8), f'episode {index}: action target vs measured joints  ({len(states)} steps)',
              fill=(245, 245, 245))
    labels = [f'j{i + 1}' for i in range(ARM_DIM)]
    for row, (data, name) in enumerate(((actions, 'action target (rad)'), (states, 'measured state (rad)'))):
        top = 34 + row * step
        _panel(draw, (left, top, right, top + panel_height),
               [data[:, j] for j in range(ARM_DIM)], COLORS,
               f'{"commanded" if row == 0 else "measured"} joints', name, labels)
    canvas.save(out_dir / f'episode_{index:06d}_joints.png')

    canvas = Image.new('RGB', (width, 44 + 2 * step), (20, 20, 23))
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 8), f'episode {index}: gripper aperture & joint tracking error',
              fill=(245, 245, 245))
    _panel(draw, (left, 34, right, 34 + panel_height),
           [actions[:, ARM_DIM], states[:, ARM_DIM]], [COLORS[0], COLORS[1]],
           'gripper aperture', 'finger width (m)', ['target', 'measured'])
    tracking = np.abs(states[:, :ARM_DIM] - actions[:, :ARM_DIM]).max(axis=1)
    top = 34 + step
    _panel(draw, (left, top, right, top + panel_height), [tracking], [COLORS[4]],
           'max |q - target| per control step', 'rad')
    canvas.save(out_dir / f'episode_{index:06d}_gripper.png')


def export_preview(result, out_dir, count=8):
    """把两路视频的抽帧拼成一张带标注的图。"""
    from PIL import Image, ImageDraw

    video_info = result['videos']
    if not video_info:
        return None
    length = result['length']
    indices = np.linspace(0, max(length - 1, 0), count).astype(int)
    rows, labels = [], []
    for key, entry in video_info.items():
        if 'frames' not in entry:
            continue
        try:
            frames = read_video_frames(entry['path'], indices)
        except Exception:  # noqa: BLE001
            continue
        row = []
        for index in indices:
            frame = frames.get(int(index))
            if frame is None:
                frame = np.zeros((256, 256, 3), dtype=np.uint8)
            row.append(Image.fromarray(frame))
        rows.append(row)
        labels.append(key.split('images.')[-1])
    if not rows:
        return None

    tile, margin, header = 128, 4, 16
    width = count * (tile + margin) + margin
    height = len(rows) * (tile + margin + header) + margin
    canvas = Image.new('RGB', (width, height), (18, 18, 20))
    draw = ImageDraw.Draw(canvas)
    for r, row in enumerate(rows):
        top = margin + r * (tile + margin + header) + header
        draw.text((margin, top - header + 2), labels[r], fill=(230, 230, 230))
        for c, image in enumerate(row):
            left = margin + c * (tile + margin)
            canvas.paste(image.resize((tile, tile), Image.BILINEAR), (left, top))
            draw.text((left + 2, top + tile - 13), f'{indices[c]}', fill=(255, 230, 120))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"episode_{result['episode_index']:06d}_preview.png"
    canvas.save(path)
    return path


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description='校验 LeRobot v2.1 数据集并做动作回放')
    parser.add_argument('--dataset', default=str(ROOT / 'data/lerobot_panda_pick_place'))
    parser.add_argument('--out', default=str(ROOT / 'results/experiment_1/validation'))
    parser.add_argument('--fps', type=int, default=20)
    parser.add_argument('--samples', type=int, default=3, help='随机抽查的 episode 数（拼图/曲线/回放）')
    parser.add_argument('--replay', type=int, default=2, help='做动作回放验证的 episode 数')
    parser.add_argument('--max-episodes', type=int, default=0, help='只检查前 N 个 episode（0 表示全部）')
    parser.add_argument('--no-video', action='store_true', help='跳过逐 episode 视频探测（仅查存在性）')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    root = Path(args.dataset)
    if not (root / 'meta/info.json').exists():
        raise SystemExit(f'找不到数据集: {root}')

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = Report()
    meta = load_meta(root)
    video_keys = check_meta(root, meta, report, expect_fps=args.fps)

    episodes = meta['episodes'][: args.max_episodes or None]
    stats_by_index = {int(row['episode_index']): row for row in meta['stats']}
    limits = joint_limits()

    results = []
    index_cursor = [0]
    for episode in episodes:
        result = check_episode(root, meta, episode, stats_by_index.get(int(episode['episode_index'])),
                               meta['tasks'], limits, video_keys, args.fps, report,
                               check_video=not args.no_video, index_cursor=index_cursor)
        results.append(result)

    # 随机抽查
    rng = random.Random(args.seed)
    usable = [row for row in results if 'states' in row]
    sampled = rng.sample(usable, min(args.samples, len(usable)))
    replay_queue = rng.sample(sampled, min(args.replay, len(sampled)))

    replay_rows = []
    for result in replay_queue:
        info = replay_episode(result['states'], result['actions'],
                              steps_per_action=int(round(240 / args.fps)))
        info['episode_index'] = result['episode_index']
        replay_rows.append(info)
        report.check(f"episode {result['episode_index']}", '动作回放关节偏差 < 0.01 rad',
                     info['mean_q_error'] < 0.01,
                     f"mean={info['mean_q_error']:.5f} max={info['max_q_error']:.5f}")

    for result in sampled:
        plot_episode(result, out / 'curves')
        export_preview(result, out / 'preview')

    # 汇总
    summary = {
        'dataset': str(root),
        'codebase_version': meta['info'].get('codebase_version'),
        'fps': meta['info'].get('fps'),
        'episodes': len(meta['episodes']),
        'frames': int(sum(int(ep['length']) for ep in meta['episodes'])),
        'checks': len(report.rows),
        'failures': len(report.failures()),
        'episode_metrics': [
            {key: value for key, value in row.items() if key not in ('states', 'actions', 'videos')}
            for row in results
        ],
        'replay': replay_rows,
        'sampled': [row['episode_index'] for row in sampled],
    }
    (out / 'report.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    report.dump_csv(out / 'checks.csv')

    if report.failures():
        print('校验失败项：')
        for row in report.failures()[:40]:
            print(f"  [{row['scope']}] {row['check']} -> {row['detail']}")
    print(f"episodes={summary['episodes']} frames={summary['frames']} "
          f"checks={summary['checks']} failures={summary['failures']}")
    if replay_rows:
        mean_err = float(np.mean([row['mean_q_error'] for row in replay_rows]))
        print(f'动作回放平均关节偏差: {mean_err:.5f} rad（{len(replay_rows)} 个 episode）')
    print(f'报告已写入: {out}')
    return 0 if not report.failures() else 1


if __name__ == '__main__':
    raise SystemExit(main())
