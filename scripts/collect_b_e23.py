"""实验 2.3 阶段 3：采集 D1（canonical）与 D2（diverse-start）B 数据。

设计（对应 prompt.md 第七阶段）：

- D1：从 HOME 出发，按 yaw 0 / 90 各 60 条采集；物体位置/朝向沿用实验 2 的
  原始 B 分布（x=0.55±12 mm、y=0.18±12 mm、yaw=0/90±0.1 rad），保证两个 yaw
  组的示范数相同；
- D2：**严格复用 D1 实际成功的 120 个物体 spec**（同一位置/朝向/种子），
  其中每个 yaw 组 30 条从 HOME、30 条从合法多样起点出发；只有机器人起点不同；
- 只存成功 episode；失败尝试与原因写入 diagnostics；D1 的失败 spec 直接换下一个，
  D2 的多样起点失败时先换起点重试、再回退 HOME（同 spec）。

用法（服务器）：

    # D1（120 条，约 35–45 分钟）
    HF_HUB_OFFLINE=1 PYTHONPATH=src venv/bin/python -u scripts/collect_b_e23.py \
        --variant D1 --episodes 120 --out data/_shards_e23/B_D1 \
        --used-out data/_shards_e23/used_specs.json

    # D2（同一 120 个 spec，一半多样起点，约 40 分钟）
    HF_HUB_OFFLINE=1 PYTHONPATH=src venv/bin/python -u scripts/collect_b_e23.py \
        --variant D2 --episodes 120 --specs data/_shards_e23/used_specs.json \
        --out data/_shards_e23/B_D2
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
sys.path.insert(0, str(ROOT / 'src'))

from manipulation import set_gripper  # noqa: E402
from simulation import HOME  # noqa: E402
from vla.atomic_tasks.config2 import (  # noqa: E402
    CAMERAS, FPS, IMAGE_SIZE, ROBOT_TYPE, STEPS_PER_ACTION, TASKS, feature_dict,
)
from vla.atomic_tasks.recording import RecordingExpert  # noqa: E402
from vla.atomic_tasks.scene2 import BLOCK_HALF, WALL_INNER_X, build_scene  # noqa: E402
from vla.atomic_tasks.starts import sample_valid_start, set_robot_state  # noqa: E402
from vla.cameras import CameraRig  # noqa: E402
from vla.lerobot_writer import LeRobotV21Writer  # noqa: E402

BLOCK_X = 0.55
X_JITTER = 0.012
Y_CENTER = 0.18
Y_JITTER = 0.012
YAW_JITTER = 0.10
PER_GROUP = 60


def depth_for_yaw(yaw: float) -> float:
    return BLOCK_HALF[0] * abs(math.cos(yaw)) + BLOCK_HALF[1] * abs(math.sin(yaw))


def make_spec(group: str, seed: int) -> dict:
    """按实验 2 的原始 B 分布生成一个物体初始状态。"""
    rng = np.random.default_rng(int(seed))
    base = 0.0 if group == 'yaw0' else math.pi / 2
    yaw = base + float(rng.uniform(-YAW_JITTER, YAW_JITTER))
    x = BLOCK_X + float(rng.uniform(-X_JITTER, X_JITTER))
    y = Y_CENTER + float(rng.uniform(-Y_JITTER, Y_JITTER))
    gap = WALL_INNER_X - (x + depth_for_yaw(yaw))
    return {
        'group': group,
        'gap_target_m': round(gap, 5),
        'yaw_rad': yaw,
        'yaw_deg': round(math.degrees(yaw), 2),
        'start': [round(x, 5), round(y, 5), 0.021],
        'seed': int(seed),
    }


def start_mode_for(index: int) -> str:
    """D2 调度：每个 yaw 组（60 条）30 条 HOME + 30 条多样起点。"""
    k = index % PER_GROUP
    if k % 2 == 1:
        return 'home'
    return ('home_neighborhood', 'home_neighborhood', 'multi_ik', 'task_relevant')[(k // 2) % 4]


def run_episode(spec: dict, index: int, variant: str, writer: LeRobotV21Writer,
                diagnostics: Path, attempt: int, mode: str) -> dict:
    started = time.time()
    sim, block, wall, tray, info = build_scene(seed=spec['seed'], start=tuple(spec['start']),
                                               jitter=0.0, yaw=spec['yaw_rad'])
    try:
        camera = CameraRig(sim, CAMERAS, IMAGE_SIZE, supersample=1)
        block_pos, _ = p.getBasePositionAndOrientation(block, physicsClientId=sim.client)
        block_pos = np.asarray(block_pos, dtype=float)
        checks = None
        rejections = {}
        if mode != 'home':
            rng = np.random.default_rng(70000000 + index * 1000 + attempt)
            cand, checks, rejections = sample_valid_start(sim, block, camera, mode, rng,
                                                          block_pos=block_pos)
            if cand is None:
                rec = {'variant': variant, 'index': index, 'attempt': attempt,
                       'mode': mode, 'sampled': False, 'rejections': rejections,
                       'wall_seconds': round(time.time() - started, 2)}
                with (diagnostics / 'episodes.jsonl').open('a', encoding='utf-8') as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
                return rec
            q = cand.q
        else:
            q = np.asarray(HOME, dtype=float)
        set_robot_state(sim, q)
        sim.command(np.asarray(q, dtype=float))
        set_gripper(sim, 0.035, 20.0, 0.05)
        for _ in range(20):
            sim.step()
        start_q = sim.q().copy()
        start_ee, _ = sim.pose()
        expert = RecordingExpert(sim, block, camera, STEPS_PER_ACTION)
        result = expert.skill_b()
        frames, states, actions = expert.episode_arrays()
        ticks = len(actions)
        record = {
            'variant': variant, 'index': index, 'attempt': attempt,
            'mode': mode, 'sampled': True, 'spec': spec,
            'scene': {k: v for k, v in info.items() if k != 'seed'},
            'block_actual': {'position': block_pos.tolist(),
                             'yaw_deg': round(float(info['block_yaw']) * 180 / math.pi, 2)},
            'start_q': start_q.tolist(),
            'start_ee': start_ee.tolist(),
            'home_distance_rad': round(float(np.linalg.norm(start_q - np.asarray(HOME))), 4),
            'checks': None if checks is None else {
                k: checks.get(k) for k in ('min_limit_margin', 'block_clearance_m',
                                           'b_approach_ik_error_m', 'camera')
            },
            'rejections': rejections,
            'success': bool(result['success']), 'reason': result['reason'],
            'metrics': result.get('metrics', {}),
            'final': result.get('final'),
            'ticks': ticks, 'seconds': round(ticks / FPS, 2),
            'wall_seconds': round(time.time() - started, 2),
        }
        if result['success'] and ticks >= 10:
            stored = writer.add_episode(frames, states, actions, TASKS['B'])
            record['episode_index'] = stored['episode_index']
        with (diagnostics / 'episodes.jsonl').open('a', encoding='utf-8') as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + '\n')
        return record
    finally:
        if p.isConnected(sim.client):
            p.disconnect(sim.client)


def collect_d1(groups: tuple[str, ...], per_group: int, out: Path, spec_seed: int,
               used_out: Path | None) -> dict:
    out = out if out.is_absolute() else ROOT / out
    if out.exists():
        import shutil
        shutil.rmtree(out)
    diagnostics = out.parent / f'_diag_D1_{out.name}'
    if diagnostics.exists():
        import shutil
        shutil.rmtree(diagnostics)
    diagnostics.mkdir(parents=True, exist_ok=True)
    writer = LeRobotV21Writer(out, feature_dict(), FPS, robot_type=ROBOT_TYPE, task=None,
                              resume=False)
    used = []
    attempts_total = 0
    started = time.time()
    for group in groups:
        seed = spec_seed + (0 if group == 'yaw0' else 50000)
        got = 0
        while got < per_group:
            spec = make_spec(group, seed)
            seed += 1
            attempts_total += 1
            rec = run_episode(spec, len(used), 'D1', writer, diagnostics, 1, 'home')
            if rec.get('success'):
                used.append(spec)
                got += 1
                print(f'[D1 {out.name} {len(used)}] {group} ok '
                      f'ticks={rec["ticks"]} gap={spec["gap_target_m"]:.3f} '
                      f'{(time.time() - started) / 60:.1f}min', flush=True)
            else:
                print(f'[D1 {out.name}] {group} skip {rec.get("reason")} '
                      f'gap={spec["gap_target_m"]:.3f}', flush=True)
    writer.finalize()
    summary = {
        'variant': 'D1', 'dataset': str(out), 'groups': list(groups),
        'episodes': len(writer.episodes), 'frames': writer.total_frames,
        'specs_used': len(used), 'attempts': attempts_total, 'seed_base': spec_seed,
        'elapsed_s': round(time.time() - started, 1), 'task': TASKS['B'],
    }
    (diagnostics / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                              encoding='utf-8')
    if used_out is not None:
        used_out = used_out if used_out.is_absolute() else ROOT / used_out
        used_out.parent.mkdir(parents=True, exist_ok=True)
        used_out.write_text(json.dumps(used, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'used specs -> {used_out} ({len(used)})')
    return summary


def collect_d2(specs: list[dict], episodes: int, out: Path) -> dict:
    out = out if out.is_absolute() else ROOT / out
    if out.exists():
        import shutil
        shutil.rmtree(out)
    diagnostics = out.parent / '_diag_D2'
    if diagnostics.exists():
        import shutil
        shutil.rmtree(diagnostics)
    diagnostics.mkdir(parents=True, exist_ok=True)
    writer = LeRobotV21Writer(out, feature_dict(), FPS, robot_type=ROBOT_TYPE, task=None,
                              resume=False)
    specs = specs[:episodes]
    attempts_total = 0
    started = time.time()
    stored = 0
    for index, spec in enumerate(specs):
        mode = start_mode_for(index)
        ok = False
        for attempt in range(1, 4):
            attempts_total += 1
            rec = run_episode(spec, index, 'D2', writer, diagnostics, attempt, mode)
            if rec.get('success'):
                ok = True
                stored += 1
                print(f'[D2 {stored}/{episodes}] idx={index} {mode} ok ticks={rec["ticks"]} '
                      f'home={rec["home_distance_rad"]:.2f} '
                      f'{(time.time() - started) / 60:.1f}min', flush=True)
                break
            print(f'[D2 {stored}/{episodes}] idx={index} {mode} fail {rec.get("reason")}',
                  flush=True)
        if not ok:
            rec = run_episode(spec, index, 'D2', writer, diagnostics, 99, 'home')
            if rec.get('success'):
                stored += 1
                print(f'[D2 {stored}/{episodes}] idx={index} HOME fallback ok', flush=True)
    writer.finalize()
    summary = {
        'variant': 'D2', 'dataset': str(out), 'episodes': len(writer.episodes),
        'frames': writer.total_frames, 'specs_used': len(specs),
        'attempts': attempts_total, 'elapsed_s': round(time.time() - started, 1),
        'task': TASKS['B'],
    }
    (diagnostics / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                              encoding='utf-8')
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=['D1', 'D2'], required=True)
    ap.add_argument('--episodes', type=int, default=120)
    ap.add_argument('--specs', type=Path, help='D2 使用的 D1 used_specs.json')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--used-out', type=Path)
    ap.add_argument('--spec-seed', type=int, default=52000000)
    ap.add_argument('--group', choices=['yaw0', 'yaw90'],
                    help='只采一个 yaw 组（用于两个进程并行）；episodes 解释为该组条数')
    args = ap.parse_args()

    if args.variant == 'D1':
        groups = (args.group,) if args.group else ('yaw0', 'yaw90')
        per_group = args.episodes if args.group else args.episodes // 2
        summary = collect_d1(groups, per_group, args.out, args.spec_seed, args.used_out)
    else:
        if args.specs is None:
            raise SystemExit('D2 需要 --specs')
        path = args.specs if args.specs.is_absolute() else ROOT / args.specs
        specs = json.loads(path.read_text(encoding='utf-8'))
        summary = collect_d2(specs, args.episodes, args.out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
