"""按 LeRobot v2.1 格式写数据集（不依赖 torch / lerobot，可在轻量仿真环境直接运行）。

目录结构与 `lerobot==0.3.3` 中 `LeRobotDataset.create/save_episode` 的输出一致：

    meta/info.json
    meta/tasks.jsonl
    meta/episodes.jsonl
    meta/episodes_stats.jsonl
    data/chunk-000/episode_000000.parquet
    videos/chunk-000/observation.images.<cam>/episode_000000.mp4

不同点仅在于本文件直接调用 ffmpeg 与 pyarrow，避免在采集端安装深度学习栈。
"""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CODEBASE_VERSION = 'v2.1'
CHUNKS_SIZE = 1000
DATA_PATH = 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet'
VIDEO_PATH = 'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4'

DEFAULT_FEATURES = {
    'timestamp': {'dtype': 'float32', 'shape': [1], 'names': None},
    'frame_index': {'dtype': 'int64', 'shape': [1], 'names': None},
    'episode_index': {'dtype': 'int64', 'shape': [1], 'names': None},
    'index': {'dtype': 'int64', 'shape': [1], 'names': None},
    'task_index': {'dtype': 'int64', 'shape': [1], 'names': None},
}


def vector_feature(names):
    return {'dtype': 'float32', 'shape': [len(names)], 'names': list(names)}


def image_feature(size, fps):
    height, width = int(size[0]), int(size[1])
    return {
        'dtype': 'video',
        'shape': [height, width, 3],
        'names': ['height', 'width', 'channels'],
        'info': {
            'video.height': height,
            'video.width': width,
            'video.codec': 'h264',
            'video.pix_fmt': 'yuv420p',
            'video.is_depth_map': False,
            'video.fps': int(fps),
            'video.channels': 3,
            'has_audio': False,
        },
    }


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=4, ensure_ascii=False), encoding='utf-8')


def _write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')


def _read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def encode_video(frames, path, fps):
    """frames: (T,H,W,3) uint8 -> H.264 / yuv420p mp4，关键帧间隔 2（与 lerobot 一致）。"""
    from imageio_ffmpeg import get_ffmpeg_exe

    frames = np.ascontiguousarray(frames, dtype=np.uint8)
    count, height, width = frames.shape[:3]
    if width % 2 or height % 2:
        raise ValueError(f'视频宽高必须为偶数，当前 {width}x{height}')
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        get_ffmpeg_exe(), '-y', '-loglevel', 'error',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}',
        '-r', str(int(fps)), '-i', '-',
        '-an', '-vcodec', 'libx264', '-pix_fmt', 'yuv420p',
        '-g', '2', '-crf', '23', '-preset', 'veryfast',
        str(path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE)
    try:
        process.stdin.write(frames.tobytes())
    finally:
        process.stdin.close()
    error = process.stderr.read().decode('utf-8', 'ignore')
    if process.wait() != 0:
        raise RuntimeError(f'ffmpeg 编码失败: {error}')


def probe_video(path):
    """用 ffmpeg 自带工具读取视频元信息，用于写入后自检。"""
    from imageio_ffmpeg import count_frames_and_secs, read_frames

    frames, seconds = count_frames_and_secs(str(path))
    generator = read_frames(str(path), pix_fmt='rgb24')
    meta = next(generator)
    generator.close()
    return {'frames': int(frames), 'seconds': float(seconds),
            'fps': float(meta['fps']), 'size': tuple(meta['size'])}


def _feature_stats(array, axis, keepdims):
    array = np.asarray(array, dtype=np.float64)
    count = int(array.shape[axis[0] if isinstance(axis, tuple) else axis])
    return {
        'min': np.min(array, axis=axis, keepdims=keepdims),
        'max': np.max(array, axis=axis, keepdims=keepdims),
        'mean': np.mean(array, axis=axis, keepdims=keepdims),
        'std': np.std(array, axis=axis, keepdims=keepdims),
        'count': np.array([count], dtype=np.int64),
    }


def _video_stats(frames, max_samples=100):
    """图像统计量：与 lerobot 相同，按 (帧, 高, 宽) 归约、除以 255，输出 (3,1,1)。

    lerobot 的 `compute_episode_stats` 最终把每通道统计量压成 (C,1,1)，
    `aggregate_stats` 会校验该形状；这里显式做 NHWC -> CHW 变换以保持一致。
    """
    array = np.asarray(frames)
    total = array.shape[0]
    indices = np.unique(np.round(np.linspace(0, total - 1, min(total, max_samples))).astype(int))
    sample = array[indices].astype(np.float32) / 255.0
    axes = (0, 1, 2)

    def reduce(func):
        value = func(sample, axis=axes, keepdims=True)  # (1,1,1,C)
        return np.squeeze(value, axis=0).transpose(2, 0, 1)  # (C,1,1)

    return {
        'min': reduce(np.min),
        'max': reduce(np.max),
        'mean': reduce(np.mean),
        'std': reduce(np.std),
        'count': np.array([total], dtype=np.int64),
    }


def _to_jsonable(stats):
    return {key: (value.tolist() if isinstance(value, np.ndarray) else value)
            for key, value in stats.items()}


class LeRobotV21Writer:
    """增量写入器：每次 add_episode 立即落盘，崩溃后可用 resume=True 继续追加。"""

    def __init__(self, root, features, fps, robot_type='unknown', task=None, resume=True,
                 video_camera_keys=None):
        self.root = Path(root)
        self.fps = int(fps)
        self.robot_type = robot_type
        self.features = {**features, **DEFAULT_FEATURES}
        self.video_keys = list(video_camera_keys) if video_camera_keys is not None else [
            key for key, feat in features.items() if feat['dtype'] == 'video']
        self.episodes = []
        self.episodes_stats = []
        self.tasks = []
        self.total_frames = 0
        if resume and (self.root / 'meta/info.json').exists():
            self._load_existing()
        else:
            self.root.mkdir(parents=True, exist_ok=True)
            (self.root / 'meta').mkdir(exist_ok=True)
        if task is not None and task not in [t['task'] for t in self.tasks]:
            self.tasks.append({'task_index': len(self.tasks), 'task': task})
        self._write_meta()

    # ---------- 元数据 ----------
    def _load_existing(self):
        info = json.loads((self.root / 'meta/info.json').read_text(encoding='utf-8'))
        if info.get('codebase_version') != CODEBASE_VERSION:
            raise ValueError(f"已有数据集的 codebase_version={info.get('codebase_version')}，"
                             f"与写入器的 {CODEBASE_VERSION} 不一致")
        if info.get('fps') != self.fps:
            raise ValueError(f"已有数据集的 fps={info.get('fps')}，与当前的 {self.fps} 不一致")
        if info.get('features') != json.loads(json.dumps(self.features)):
            raise ValueError('已有数据集的 features 与当前定义不一致，无法续写')
        self.episodes = _read_jsonl(self.root / 'meta/episodes.jsonl')
        self.episodes_stats = _read_jsonl(self.root / 'meta/episodes_stats.jsonl')
        self.tasks = _read_jsonl(self.root / 'meta/tasks.jsonl')
        self.total_frames = int(sum(ep['length'] for ep in self.episodes))

    def _info(self):
        total = len(self.episodes)
        info = {
            'codebase_version': CODEBASE_VERSION,
            'robot_type': self.robot_type,
            'total_episodes': total,
            'total_frames': self.total_frames,
            'total_tasks': len(self.tasks),
            'total_videos': total * len(self.video_keys),
            'total_chunks': int(np.ceil(total / CHUNKS_SIZE)) if total else 0,
            'chunks_size': CHUNKS_SIZE,
            'fps': self.fps,
            'splits': {'train': f'0:{total}'},
            'data_path': DATA_PATH,
            'video_path': VIDEO_PATH,
            'features': self.features,
        }
        return info

    def _write_meta(self):
        _write_json(self.root / 'meta/info.json', self._info())
        _write_jsonl(self.root / 'meta/tasks.jsonl', self.tasks)
        _write_jsonl(self.root / 'meta/episodes.jsonl', self.episodes)
        _write_jsonl(self.root / 'meta/episodes_stats.jsonl', self.episodes_stats)

    # ---------- episode ----------
    def _task_index(self, task):
        for item in self.tasks:
            if item['task'] == task:
                return item['task_index']
        index = len(self.tasks)
        self.tasks.append({'task_index': index, 'task': task})
        return index

    def _resolve_frames(self, frames):
        """允许 frames 用完整特征名（observation.images.top）或短名（top）作键。"""
        resolved = {}
        for key in self.video_keys:
            if key in frames:
                resolved[key] = frames[key]
            else:
                short = key.rsplit('.', 1)[-1]
                if short in frames:
                    resolved[key] = frames[short]
        return resolved

    def add_episode(self, frames, states, actions, task, stage_labels=None, check_video=True):
        """frames: {cam: (T,H,W,3) uint8}；states: (T,S)；actions: (T,A)。"""
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        if states.ndim != 2 or actions.ndim != 2:
            raise ValueError('states/actions 必须是二维数组')
        length = len(actions)
        if length < 2 or len(states) != length:
            raise ValueError(f'长度不合法: states={len(states)}, actions={length}')
        if not np.all(np.isfinite(states)) or not np.all(np.isfinite(actions)):
            raise ValueError('states/actions 含 NaN 或 Inf')
        expected_state = self.features['observation.state']['shape'][0]
        expected_action = self.features['action']['shape'][0]
        if states.shape[1] != expected_state or actions.shape[1] != expected_action:
            raise ValueError(f'维度不符: state {states.shape[1]}!={expected_state}, '
                             f'action {actions.shape[1]}!={expected_action}')
        if stage_labels is not None and len(stage_labels) != length:
            raise ValueError('stage_labels 长度与帧数不一致')
        videos = self._resolve_frames(frames)
        for key in self.video_keys:
            if key not in videos:
                raise ValueError(f'缺少相机 {key} 的帧，已有 {list(frames)}')
            expected_shape = self.features[key]['shape']
            array = np.asarray(videos[key])
            if array.shape != (length, *expected_shape):
                raise ValueError(f'{key} 形状 {array.shape} 与 features {expected_shape} / T={length} 不符')
            if array.dtype != np.uint8:
                raise ValueError(f'{key} 必须是 uint8，当前 {array.dtype}')

        episode_index = len(self.episodes)
        chunk = episode_index // CHUNKS_SIZE
        task_index = self._task_index(task)
        frame_index = np.arange(length, dtype=np.int64)
        timestamps = (frame_index / float(self.fps)).astype(np.float32)
        global_index = np.arange(self.total_frames, self.total_frames + length, dtype=np.int64)

        episode = {'episode_index': episode_index, 'tasks': [task], 'length': length}

        table = pa.table({
            'observation.state': pa.array([row for row in states], type=pa.list_(pa.float32())),
            'action': pa.array([row for row in actions], type=pa.list_(pa.float32())),
            'timestamp': pa.array(timestamps, type=pa.float32()),
            'frame_index': pa.array(frame_index, type=pa.int64()),
            'episode_index': pa.array(np.full(length, episode_index, dtype=np.int64), type=pa.int64()),
            'index': pa.array(global_index, type=pa.int64()),
            'task_index': pa.array(np.full(length, task_index, dtype=np.int64), type=pa.int64()),
        })
        data_file = self.root / DATA_PATH.format(episode_chunk=chunk, episode_index=episode_index)
        data_file.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, data_file, compression='snappy')

        video_stats = {}
        for key in self.video_keys:
            video_file = self.root / VIDEO_PATH.format(episode_chunk=chunk, video_key=key,
                                                       episode_index=episode_index)
            encode_video(videos[key], video_file, self.fps)
            if check_video:
                meta = probe_video(video_file)
                if meta['frames'] != length:
                    raise RuntimeError(f'{video_file} 帧数 {meta["frames"]} != {length}')
                if abs(meta['fps'] - self.fps) > 1e-3:
                    raise RuntimeError(f'{video_file} 帧率 {meta["fps"]} != {self.fps}')
            video_stats[key] = _video_stats(videos[key])

        stats = {
            'observation.state': _to_jsonable(_feature_stats(states, axis=0, keepdims=False)),
            'action': _to_jsonable(_feature_stats(actions, axis=0, keepdims=False)),
            'timestamp': _to_jsonable(_feature_stats(timestamps, axis=0, keepdims=True)),
            'frame_index': _to_jsonable(_feature_stats(frame_index, axis=0, keepdims=True)),
            'episode_index': _to_jsonable(_feature_stats(
                np.full(length, episode_index, dtype=np.float64), axis=0, keepdims=True)),
            'index': _to_jsonable(_feature_stats(global_index.astype(np.float64), axis=0, keepdims=True)),
            'task_index': _to_jsonable(_feature_stats(
                np.full(length, task_index, dtype=np.float64), axis=0, keepdims=True)),
        }
        for key, value in video_stats.items():
            stats[key] = _to_jsonable(value)

        self.episodes.append(episode)
        self.episodes_stats.append({'episode_index': episode_index, 'stats': stats})
        self.total_frames += length
        self._write_meta()
        return {'episode_index': episode_index, 'length': length,
                'data_file': str(data_file), 'chunk': chunk}

    def finalize(self):
        self._write_meta()
        return self._info()

    # ---------- 自检 ----------
    def verify(self):
        """对已落盘的每一集做尺寸/时间戳/视频帧数一致性检查。"""
        reports = []
        for episode in self.episodes:
            index = episode['episode_index']
            chunk = index // CHUNKS_SIZE
            data_file = self.root / DATA_PATH.format(episode_chunk=chunk, episode_index=index)
            table = pq.read_table(data_file)
            length = table.num_rows
            timestamps = table.column('timestamp').to_numpy()
            expected = np.arange(length) / self.fps
            ok = (length == episode['length']
                  and np.allclose(timestamps, expected, atol=1e-4)
                  and table.column('episode_index').to_pylist() == [index] * length
                  and table.column('frame_index').to_pylist() == list(range(length)))
            videos = {}
            for key in self.video_keys:
                video_file = self.root / VIDEO_PATH.format(episode_chunk=chunk, video_key=key,
                                                           episode_index=index)
                videos[key] = probe_video(video_file)
                ok = ok and videos[key]['frames'] == length
            reports.append({'episode_index': index, 'ok': bool(ok), 'length': length,
                            'videos': videos})
        return reports


def merge_datasets(shards, out_root, features, fps, robot_type, task, verbose=True):
    """把多个 shard 数据集合并成一个（视频直接复制、parquet 重新编号）。

    并行采集时每个进程写自己的 shard，合并后 episode_index / index 连续，
    其余统计量沿用 shard 内已算好的结果，只修正与序号相关的三项。
    """
    shards = [Path(s) for s in shards]
    writer = LeRobotV21Writer(out_root, features, fps, robot_type=robot_type, task=task,
                              resume=False)
    episodes, episodes_stats = [], []
    total_frames = 0
    for shard in shards:
        info_file = shard / 'meta/info.json'
        if not info_file.exists():
            continue
        info = json.loads(info_file.read_text(encoding='utf-8'))
        available = {key: value for key, value in info['features'].items()
                     if key not in DEFAULT_FEATURES}
        if info['fps'] != fps or available != json.loads(json.dumps(features)):
            raise ValueError(f'{shard} 的 fps/features 与目标数据集不一致')
        shard_episodes = _read_jsonl(shard / 'meta/episodes.jsonl')
        shard_stats = {row['episode_index']: row['stats']
                       for row in _read_jsonl(shard / 'meta/episodes_stats.jsonl')}
        for episode in shard_episodes:
            source_index = episode['episode_index']
            chunk = source_index // CHUNKS_SIZE
            source_table = pq.read_table(shard / DATA_PATH.format(episode_chunk=chunk,
                                                                  episode_index=source_index))
            length = source_table.num_rows
            target_index = len(episodes)
            global_start = total_frames
            target_chunk = target_index // CHUNKS_SIZE
            table = pa.table({
                'observation.state': source_table.column('observation.state'),
                'action': source_table.column('action'),
                'timestamp': source_table.column('timestamp'),
                'frame_index': source_table.column('frame_index'),
                'episode_index': pa.array(np.full(length, target_index, dtype=np.int64),
                                          type=pa.int64()),
                'index': pa.array(np.arange(global_start, global_start + length, dtype=np.int64),
                                  type=pa.int64()),
                'task_index': pa.array(np.zeros(length, dtype=np.int64), type=pa.int64()),
            })
            target_data = writer.root / DATA_PATH.format(episode_chunk=target_chunk,
                                                         episode_index=target_index)
            target_data.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, target_data, compression='snappy')

            for key in writer.video_keys:
                source_video = shard / VIDEO_PATH.format(episode_chunk=chunk, video_key=key,
                                                         episode_index=source_index)
                target_video = writer.root / VIDEO_PATH.format(episode_chunk=target_chunk,
                                                               video_key=key,
                                                               episode_index=target_index)
                target_video.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_video, target_video)
                if probe_video(target_video)['frames'] != length:
                    raise RuntimeError(f'{target_video} 帧数与 parquet 不一致')

            stats = {key: dict(value) for key, value in shard_stats[source_index].items()}
            stats['episode_index'] = {'min': [target_index], 'max': [target_index],
                                      'mean': [target_index], 'std': [0.0], 'count': [length]}
            stats['task_index'] = {'min': [0], 'max': [0], 'mean': [0], 'std': [0.0],
                                   'count': [length]}
            index_values = np.arange(global_start, global_start + length, dtype=np.float64)
            stats['index'] = {'min': [float(index_values.min())], 'max': [float(index_values.max())],
                              'mean': [float(index_values.mean())], 'std': [float(index_values.std())],
                              'count': [length]}
            episodes.append({'episode_index': target_index, 'tasks': [task], 'length': length})
            episodes_stats.append({'episode_index': target_index, 'stats': stats})
            total_frames += length
            if verbose and target_index % 10 == 0:
                print(f'  merge ep {target_index} ({shard.name}) length={length}')

    writer.episodes = episodes
    writer.episodes_stats = episodes_stats
    writer.total_frames = total_frames
    info = writer.finalize()
    if verbose:
        print(f'合并完成: {len(episodes)} 个 episode, {total_frames} 帧 -> {writer.root}')
    return info


def merge_multi_task(shards, out_root, features, fps, robot_type, verbose=True):
    """合并多个单任务 shard，并保留每集自己的任务文本与 task_index。

    与 merge_datasets 的区别：任务不再是全局单一字符串，而是按 episode 保留。
    """
    shards = [Path(s) for s in shards]
    writer = LeRobotV21Writer(out_root, features, fps, robot_type=robot_type, task=None,
                              resume=False)
    episodes, episodes_stats = [], []
    total_frames = 0
    for shard in shards:
        info_file = shard / 'meta/info.json'
        if not info_file.exists():
            continue
        info = json.loads(info_file.read_text(encoding='utf-8'))
        available = {key: value for key, value in info['features'].items()
                     if key not in DEFAULT_FEATURES}
        if info['fps'] != fps or available != json.loads(json.dumps(features)):
            raise ValueError(f'{shard} 的 fps/features 与目标数据集不一致')
        shard_episodes = _read_jsonl(shard / 'meta/episodes.jsonl')
        shard_stats = {row['episode_index']: row['stats']
                       for row in _read_jsonl(shard / 'meta/episodes_stats.jsonl')}
        for episode in shard_episodes:
            source_index = episode['episode_index']
            chunk = source_index // CHUNKS_SIZE
            source_table = pq.read_table(shard / DATA_PATH.format(episode_chunk=chunk,
                                                                  episode_index=source_index))
            length = source_table.num_rows
            task = episode['tasks'][0]
            target_index = len(episodes)
            global_start = total_frames
            target_chunk = target_index // CHUNKS_SIZE
            task_index = writer._task_index(task)
            table = pa.table({
                'observation.state': source_table.column('observation.state'),
                'action': source_table.column('action'),
                'timestamp': source_table.column('timestamp'),
                'frame_index': source_table.column('frame_index'),
                'episode_index': pa.array(np.full(length, target_index, dtype=np.int64),
                                          type=pa.int64()),
                'index': pa.array(np.arange(global_start, global_start + length, dtype=np.int64),
                                  type=pa.int64()),
                'task_index': pa.array(np.full(length, task_index, dtype=np.int64),
                                       type=pa.int64()),
            })
            target_data = writer.root / DATA_PATH.format(episode_chunk=target_chunk,
                                                         episode_index=target_index)
            target_data.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, target_data, compression='snappy')

            for key in writer.video_keys:
                source_video = shard / VIDEO_PATH.format(episode_chunk=chunk, video_key=key,
                                                         episode_index=source_index)
                target_video = writer.root / VIDEO_PATH.format(episode_chunk=target_chunk,
                                                               video_key=key,
                                                               episode_index=target_index)
                target_video.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_video, target_video)
                if probe_video(target_video)['frames'] != length:
                    raise RuntimeError(f'{target_video} 帧数与 parquet 不一致')

            stats = {key: dict(value) for key, value in shard_stats[source_index].items()}
            stats['episode_index'] = {'min': [target_index], 'max': [target_index],
                                      'mean': [target_index], 'std': [0.0], 'count': [length]}
            stats['task_index'] = {'min': [task_index], 'max': [task_index],
                                   'mean': [task_index], 'std': [0.0], 'count': [length]}
            index_values = np.arange(global_start, global_start + length, dtype=np.float64)
            stats['index'] = {'min': [float(index_values.min())], 'max': [float(index_values.max())],
                              'mean': [float(index_values.mean())],
                              'std': [float(index_values.std())], 'count': [length]}
            episodes.append({'episode_index': target_index, 'tasks': [task], 'length': length})
            episodes_stats.append({'episode_index': target_index, 'stats': stats})
            total_frames += length
            if verbose and target_index % 20 == 0:
                print(f'  merge ep {target_index} ({shard.name}) task={task[:32]} '
                      f'length={length}')

    writer.episodes = episodes
    writer.episodes_stats = episodes_stats
    writer.total_frames = total_frames
    info = writer.finalize()
    if verbose:
        print(f'合并完成: {len(episodes)} 个 episode, {total_frames} 帧, '
              f'{len(writer.tasks)} 个任务 -> {writer.root}')
    return info


def delete_dataset(root):
    root = Path(root)
    if root.exists():
        shutil.rmtree(root)
