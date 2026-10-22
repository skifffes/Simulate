"""实验 2 阶段 2：按原子技能采集专家示范（A0 / A90 / B / C）。

每个技能单独采集到一个 shard，最后用 merge_multi_task 合成一个多任务数据集。
训练集只包含独立原子任务，不含任何 A→B / A→C 连续轨迹。

用法：
    # 冒烟：每类 3 条
    python scripts/collect_multi_atomic_data.py --skill A0 --episodes 3 --out data/_shards_exp2/A0

    # 正式：每类 120 条（可四个进程并行）
    python scripts/collect_multi_atomic_data.py --skill B --episodes 120 --out data/_shards_exp2/B

    # 合并
    python scripts/collect_multi_atomic_data.py --merge --out data/lerobot_panda_multi \
        --shards data/_shards_exp2
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pybullet as p

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from vla.atomic_tasks.config2 import (  # noqa: E402
    CAMERAS, FPS, IMAGE_SIZE, ROBOT_TYPE, STEPS_PER_ACTION, TASKS, feature_dict,
)
from vla.atomic_tasks.recording import RecordingExpert  # noqa: E402
from vla.atomic_tasks.scene2 import PLACE, build_scene  # noqa: E402
from vla.cameras import CameraRig  # noqa: E402
from vla.lerobot_writer import LeRobotV21Writer, merge_multi_task  # noqa: E402

SKILL_FN = {'A0': 'skill_a0', 'A90': 'skill_a90', 'B': 'skill_b', 'C': 'skill_c'}


def sample_scene(skill: str, seed: int) -> dict:
    """每个技能自己的初始状态分布。

    A0/A90 从标准起点开始（长轴 Y）；B/C 从“类 A 终态”的放置区开始，
    朝向取长轴 Y 或 X（各半），位置扰动 ±12 mm。
    """
    rng = np.random.default_rng(int(seed))
    if skill in ('A0', 'A90'):
        return {'jitter': 0.008, 'yaw': 0.0}
    base_yaw = 0.0 if rng.random() < 0.5 else math.pi / 2
    yaw = base_yaw + float(rng.uniform(-0.10, 0.10))
    return {'start': PLACE, 'jitter': 0.012, 'yaw': yaw}


def run_episode(skill: str, seed: int, out: Path, writer: LeRobotV21Writer,
                diagnostics: Path) -> dict:
    scene_kw = sample_scene(skill, seed)
    started = time.time()
    sim, block, wall, tray, info = build_scene(seed=seed, **scene_kw)
    try:
        camera = CameraRig(sim, CAMERAS, IMAGE_SIZE, supersample=1)
        expert = RecordingExpert(sim, block, camera, STEPS_PER_ACTION)
        result = getattr(expert, SKILL_FN[skill])()
        frames, states, actions = expert.episode_arrays()
        ticks = len(actions)
        record = {
            'skill': skill, 'seed': seed, 'scene': info,
            'success': bool(result['success']), 'reason': result['reason'],
            'ticks': ticks, 'seconds': ticks / FPS,
            'metrics': result.get('metrics', {}),
            'final': result.get('final'),
            'wall_seconds': round(time.time() - started, 2),
        }
        if result['success']:
            if ticks < 10:
                raise RuntimeError(f'{skill} episode 只有 {ticks} 个控制周期')
            if not writer.episodes:
                from PIL import Image
                preview = diagnostics / 'preview'
                preview.mkdir(parents=True, exist_ok=True)
                for cam, arr in frames.items():
                    for tag, pick in (('first', 0), ('mid', ticks // 2), ('last', ticks - 1)):
                        Image.fromarray(arr[pick]).save(preview / f'{skill}_{cam}_{tag}.png')
            stored = writer.add_episode(frames, states, actions, TASKS[skill])
            record['episode_index'] = stored['episode_index']
        with (diagnostics / 'episodes.jsonl').open('a', encoding='utf-8') as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + '\n')
        return record
    finally:
        if p.isConnected(sim.client):
            p.disconnect(sim.client)


def collect(skill: str, episodes: int, out: Path, seed_base: int, max_attempts: int,
            overwrite: bool):
    out = out if out.is_absolute() else ROOT / out
    if overwrite and out.exists():
        import shutil
        shutil.rmtree(out)
    diagnostics = out.parent / f'_diag_{out.name}'
    diagnostics.mkdir(parents=True, exist_ok=True)
    writer = LeRobotV21Writer(out, feature_dict(), FPS, robot_type=ROBOT_TYPE, task=None,
                              resume=True)
    done = len(writer.episodes)
    attempts = 0
    stored = done
    started = time.time()
    while stored < episodes and attempts < max_attempts:
        attempts += 1
        seed = int(seed_base) + done * 100 + attempts
        record = run_episode(skill, seed, out, writer, diagnostics)
        if record['success']:
            stored += 1
        elapsed = time.time() - started
        rate = stored / max(elapsed, 1e-6) * 60
        eta = (episodes - stored) / max(rate, 1e-6)
        print(f'[{skill} {stored}/{episodes}] seed={seed} success={record["success"]} '
              f'reason={record["reason"]} ticks={record["ticks"]} '
              f'{elapsed / 60:.1f}min 预计剩余 {eta:.1f}min', flush=True)
    writer.finalize()
    summary = {
        'skill': skill, 'dataset': str(out), 'episodes': len(writer.episodes),
        'frames': writer.total_frames, 'attempts': attempts,
        'seed_base': seed_base, 'elapsed_s': round(time.time() - started, 1),
        'task': TASKS[skill],
    }
    (diagnostics / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                              encoding='utf-8')
    return summary


def merge(shards: Path, out: Path):
    shards = shards if shards.is_absolute() else ROOT / shards
    out = out if out.is_absolute() else ROOT / out
    dirs = sorted(p for p in shards.iterdir() if (p / 'meta/info.json').exists())
    info = merge_multi_task(dirs, out, feature_dict(), FPS, ROBOT_TYPE)
    print(json.dumps(info, indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skill', choices=['A0', 'A90', 'B', 'C'])
    ap.add_argument('--episodes', type=int, default=3)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seed-base', type=int, default=900000)
    ap.add_argument('--max-attempts', type=int, default=0)
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--merge', action='store_true')
    ap.add_argument('--shards', type=Path)
    args = ap.parse_args()

    if args.merge:
        if args.shards is None:
            raise SystemExit('--merge 需要 --shards')
        merge(args.shards, args.out)
        return
    if args.skill is None:
        raise SystemExit('需要 --skill 或 --merge')
    total = args.max_attempts or max(int(args.episodes * 1.5) + 5, args.episodes + 5)
    summary = collect(args.skill, args.episodes, args.out, args.seed_base, total,
                      args.overwrite)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
