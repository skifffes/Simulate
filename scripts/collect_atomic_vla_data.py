"""采集实验 1 的专家演示并写成 LeRobot v2.1 数据集（支持多进程分片采集后合并）。

用法（在仓库根目录）：
    .conda312\\python.exe scripts\\collect_atomic_vla_data.py --episodes 10 --workers 5 --out data\\lerobot_panda_pick_place_smoke
    .conda312\\python.exe scripts\\collect_atomic_vla_data.py --episodes 150 --workers 6

每个 episode 重新构建场景并小范围随机化（工件平面位置、朝向、摩擦）；只有通过
0C 全部成功判定的 episode 才写入数据集，失败尝试写入诊断日志。
"""
import argparse
import json
import multiprocessing as mp
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pybullet as p

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from robot_controller import RobotController  # noqa: E402
from vla.cameras import CameraRig  # noqa: E402
from vla.config import (CAMERAS, DATASET_DIRNAME, FPS, IMAGE_SIZE, ROBOT_TYPE,  # noqa: E402
                        STEPS_PER_ACTION, TASK, feature_dict)
from vla.expert import VLAExpertTask  # noqa: E402
from vla.lerobot_writer import LeRobotV21Writer, merge_datasets  # noqa: E402
from vla.scenario import build_scene, scenario_for_seed  # noqa: E402

DIAGNOSTICS = ROOT / 'results' / 'experiment_1' / 'diagnostics'


def save_preview(frames, out_dir, index):
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    for camera, array in frames.items():
        for tag, pick in (('first', 0), ('mid', len(array) // 2), ('last', len(array) - 1)):
            Image.fromarray(array[pick]).save(out_dir / f'ep{index:03d}_{camera}_{tag}.png')


def _jsonable(value):
    if isinstance(value, tuple):
        return list(value)
    return value


def run_collection(out, episodes, seed, jitter, friction_jitter, supersample,
                   max_attempts, overwrite, no_preview, diag_dir, label=''):
    """单进程采集，返回汇总 dict。"""
    if overwrite and out.exists():
        shutil.rmtree(out)
    diag_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = diag_dir / 'preview'
    features = feature_dict()
    writer = LeRobotV21Writer(out, features, FPS, robot_type=ROBOT_TYPE, task=TASK,
                              resume=not overwrite)
    if writer.episodes:
        print(f'{label}[resume] 已有 {len(writer.episodes)} 个 episode，{writer.total_frames} 帧')

    log_path = diag_dir / 'collection_log.csv'
    log_exists = log_path.exists()
    log_fields = ['attempt', 'seed', 'episode_index', 'stored', 'success', 'failure', 'failure_stage',
                  'ticks', 'seconds', 'offset_m', 'block_friction', 'block_yaw',
                  'place_xy_error_m', 'tilt_deg', 'start_time']
    log_file = log_path.open('a', newline='', encoding='utf-8')
    import csv
    log = csv.DictWriter(log_file, fieldnames=log_fields)
    if not log_exists:
        log.writeheader()

    stored = 0
    attempts = 0
    started = time.time()
    try:
        while stored < episodes and attempts < max_attempts:
            attempts += 1
            episode_seed = int(seed) + attempts
            scenario = scenario_for_seed(episode_seed, jitter, friction_jitter)
            params, offset = scenario.params, scenario.offset_m
            sim, block = build_scene(scenario)
            try:
                camera = CameraRig(sim, CAMERAS, IMAGE_SIZE, supersample=supersample)
                task = VLAExpertTask(sim, RobotController(sim), block, params=params, camera=camera)
                success = task.run()
                result = task.result()
                frames, states, actions, labels = task.episode_arrays()
                stacked = {cam: np.stack([frame[cam] for frame in frames]) for cam in CAMERAS}
                ticks = len(actions)
                episode_index = ''
                if success:
                    if ticks < 10:
                        raise RuntimeError(f'episode 只有 {ticks} 个控制周期，异常')
                    info = writer.add_episode(stacked, states, actions, TASK, stage_labels=labels)
                    episode_index = info['episode_index']
                    stored += 1
                    if not no_preview and stored <= 3:
                        save_preview(stacked, preview_dir, stored - 1)
                    with (diag_dir / 'episodes.jsonl').open('a', encoding='utf-8') as fh:
                        fh.write(json.dumps({
                            'episode_index': int(episode_index), 'seed': episode_seed,
                            'ticks': ticks, 'seconds': ticks / FPS, 'offset_m': offset,
                            'block_yaw': scenario.block_yaw,
                            'block_friction': scenario.block_friction,
                            'params': {k: _jsonable(v) for k, v in params.items()},
                            'result': result,
                        }, ensure_ascii=False) + '\n')
                else:
                    with (diag_dir / 'failures.jsonl').open('a', encoding='utf-8') as fh:
                        fh.write(json.dumps({
                            'seed': episode_seed, 'offset_m': offset,
                            'params': {k: _jsonable(v) for k, v in params.items()},
                            'result': result}, ensure_ascii=False) + '\n')
                log.writerow({
                    'attempt': attempts, 'seed': episode_seed,
                    'episode_index': episode_index, 'stored': bool(success),
                    'success': bool(success), 'failure': result['failure'],
                    'failure_stage': result['failure_stage'], 'ticks': ticks,
                    'seconds': f'{ticks / FPS:.3f}', 'offset_m': f'{offset:.4f}',
                    'block_friction': f'{params["block_friction"]:.3f}',
                    'block_yaw': f'{params.get("block_yaw", 0.0):.3f}',
                    'place_xy_error_m': f'{result["place_xy_error_m"]:.4f}',
                    'tilt_deg': f'{result["final_tilt_deg"]:.2f}',
                    'start_time': time.strftime('%H:%M:%S'),
                })
                log_file.flush()
                elapsed = time.time() - started
                eta = elapsed / max(attempts, 1) * (max_attempts - attempts)
                print(f'{label}[{stored}/{episodes}] attempt={attempts} success={success} '
                      f'failure={result["failure"]} ticks={ticks} 已用 {elapsed / 60:.1f}min '
                      f'剩余约 {eta / 60:.1f}min', flush=True)
            finally:
                if p.isConnected(sim.client):
                    p.disconnect(sim.client)
    except KeyboardInterrupt:
        print('\n中断：已写入的 episode 保留在数据集中')
    finally:
        log_file.close()

    info = writer.finalize()
    summary = {
        'dataset': str(out), 'episodes': info['total_episodes'], 'frames': info['total_frames'],
        'fps': info['fps'], 'attempts': attempts, 'stored': stored,
        'elapsed_s': round(time.time() - started, 1),
        'size_mb': round(sum(f.stat().st_size for f in out.rglob('*') if f.is_file()) / 2 ** 20, 1)
        if out.exists() else 0.0,
        'seed': seed, 'jitter_m': jitter, 'friction_jitter': friction_jitter,
        'steps_per_action': STEPS_PER_ACTION,
    }
    (diag_dir / 'collection_summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    return summary


def _worker(payload):
    return run_collection(**payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--out', type=Path, default=ROOT / DATASET_DIRNAME)
    parser.add_argument('--seed', type=int, default=20261021)
    parser.add_argument('--jitter', type=float, default=0.008, help='工件平面位置随机幅度 (m)')
    parser.add_argument('--friction-jitter', type=float, default=0.2)
    parser.add_argument('--supersample', type=int, default=1)
    parser.add_argument('--max-attempts', type=int, default=0,
                        help='最大尝试次数，默认为目标 episode 数的 1.6 倍')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--no-preview', action='store_true')
    parser.add_argument('--clean-shards', action='store_true', help='合并后删除分片目录')
    args = parser.parse_args()

    out = args.out if args.out.is_absolute() else ROOT / args.out
    total_attempts = args.max_attempts or max(int(args.episodes * 1.6) + 4, args.episodes + 4)
    DIAGNOSTICS.mkdir(parents=True, exist_ok=True)

    if args.workers <= 1:
        summary = run_collection(out, args.episodes, args.seed, args.jitter,
                                 args.friction_jitter, args.supersample, total_attempts,
                                 args.overwrite, args.no_preview, DIAGNOSTICS)
        (DIAGNOSTICS / 'collection_summary.json').write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    workers = args.workers
    shard_root = out.parent / f'_shards_{out.name}'
    if shard_root.exists():
        shutil.rmtree(shard_root)
    per_worker = int(np.ceil(args.episodes / workers))
    payloads = []
    for index in range(workers):
        payloads.append({
            'out': shard_root / f'shard_{index:02d}',
            'episodes': per_worker,
            'seed': args.seed + index * 10007,
            'jitter': args.jitter,
            'friction_jitter': args.friction_jitter,
            'supersample': args.supersample,
            'max_attempts': int(np.ceil(total_attempts / workers)) + 2,
            'overwrite': True,
            'no_preview': args.no_preview,
            'diag_dir': DIAGNOSTICS / f'shard_{index:02d}',
            'label': f'[w{index}] ',
        })
    started = time.time()
    context = mp.get_context('spawn')
    with context.Pool(workers) as pool:
        results = pool.map(_worker, payloads)

    shards = [payload['out'] for payload in payloads]
    info = merge_datasets(shards, out, feature_dict(), FPS, ROBOT_TYPE, TASK)
    total_attempts_used = sum(r['attempts'] for r in results)
    summary = {
        'dataset': str(out), 'episodes': info['total_episodes'], 'frames': info['total_frames'],
        'fps': info['fps'], 'attempts': total_attempts_used,
        'stored': sum(r['stored'] for r in results),
        'elapsed_s': round(time.time() - started, 1),
        'size_mb': round(sum(f.stat().st_size for f in out.rglob('*') if f.is_file()) / 2 ** 20, 1),
        'seed': args.seed, 'jitter_m': args.jitter, 'friction_jitter': args.friction_jitter,
        'workers': workers, 'steps_per_action': STEPS_PER_ACTION,
        'per_shard': results,
    }
    shutil.copy2(shards[0] / 'meta/info.json', DIAGNOSTICS / 'info_shard0.json')
    (DIAGNOSTICS / 'collection_summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    if args.clean_shards:
        shutil.rmtree(shard_root)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
