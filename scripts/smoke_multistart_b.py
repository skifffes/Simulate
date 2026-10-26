"""实验 2.3 阶段 2：B 原子专家的多起点冒烟验证。

对每个 episode：
1. 用 starts.sample_valid_start 采一个物理合法的机器人起始构型（三种模式轮换）；
2. 只在 episode 开始时把机器人初始化到该构型（不瞬移、不在执行中重置）；
3. 直接运行 B 原子专家（不改任务目标、不先回 HOME）；
4. 记录成功/失败、失败原因、关节逐步最大跳变、物体抬升、路径与耗时。

输出：
    results/experiment_2_3/smoke/starts.jsonl     每个接受起点的合法性证据 + 被拒原因统计
    results/experiment_2_3/smoke/episodes.jsonl   逐 episode 结果（含专家 trace）
    results/experiment_2_3/smoke/episodes.csv     扁平表
    results/experiment_2_3/smoke/summary.json     分模式 / 分 yaw 汇总
    results/experiment_2_3/smoke/preview/*.png    代表性起点与终态画面

用法（服务器）：
    HF_HUB_OFFLINE=1 PYTHONPATH=src venv/bin/python -u \
        scripts/smoke_multistart_b.py --episodes 36 --seed-base 50000000 \
        --out results/experiment_2_3/smoke
"""
from __future__ import annotations

import argparse
import csv
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
from robot_controller import MotionResult  # noqa: E402
from simulation import HOME  # noqa: E402
from vla.atomic_tasks.config2 import CAMERAS, IMAGE_SIZE, STEPS_PER_ACTION  # noqa: E402
from vla.atomic_tasks.experts import SkillExpert  # noqa: E402
from vla.atomic_tasks.scene2 import PLACE, build_scene, grasp_quat  # noqa: E402
from vla.atomic_tasks.starts import (  # noqa: E402
    LIMIT_MARGIN, sample_valid_start, set_robot_state,
)
from vla.cameras import CameraRig  # noqa: E402

MODES = ('home', 'home_neighborhood', 'multi_ik', 'task_relevant')


class MonitoredExpert(SkillExpert):
    """在专家执行期间记录关节跳变、到 HOME 的距离、物体高度轨迹。"""

    def __init__(self, sim, block):
        super().__init__(sim, block)
        self.max_joint_step = 0.0
        self.home_min = float('inf')
        self.max_obj_z = 0.0
        self.obj_trace = []
        self._last_q = np.asarray(sim.q(), dtype=float).copy()

    def _observe(self, k, n, q, seconds):
        q = np.asarray(q, dtype=float)
        self.max_joint_step = max(self.max_joint_step, float(np.max(np.abs(q - self._last_q))))
        self.home_min = min(self.home_min, float(np.linalg.norm(q - np.asarray(HOME))))
        self._last_q = q.copy()
        pos, _ = p.getBasePositionAndOrientation(self.block, physicsClientId=self.sim.client)
        self.max_obj_z = max(self.max_obj_z, float(pos[2]))
        self.obj_trace.append([round(seconds, 4), round(float(pos[0]), 5),
                               round(float(pos[1]), 5), round(float(pos[2]), 5)])

    def _move(self, pos, yaw, tag):
        target = np.asarray(pos, dtype=float)
        q, ok, _ = self.ctl.solve(target, grasp_quat(yaw))
        entry = {'tag': tag, 'target': target.tolist(), 'yaw': float(yaw), 'ok': bool(ok)}
        if not ok:
            entry['reason'] = 'ik_joint_limits'
            self.trace.append(entry)
            return MotionResult(False, 'ik_joint_limits', 0, 0, False, False, False, False, None)
        result = self.ctl.move(q, observer=self._observe)
        entry['ok'] = bool(result.success)
        entry['reason'] = result.reason
        self.trace.append(entry)
        return result


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def run_episode(mode: str, seed: int, out: Path, preview: bool):
    rng = np.random.default_rng(seed)
    yaw0 = 0.0 if rng.random() < 0.5 else math.pi / 2
    yaw = yaw0 + float(rng.uniform(-0.10, 0.10))
    sim, block, wall, tray, info = build_scene(seed=seed, start=PLACE, jitter=0.012, yaw=yaw)
    started = time.time()
    try:
        camera = CameraRig(sim, CAMERAS, IMAGE_SIZE, supersample=1)
        block_pos, _ = p.getBasePositionAndOrientation(block, physicsClientId=sim.client)
        block_pos = np.asarray(block_pos, dtype=float)
        cand, checks, rejections = sample_valid_start(sim, block, camera, mode, rng,
                                                      block_pos=block_pos)
        if cand is None:
            return {'mode': mode, 'seed': seed, 'sampled': False,
                    'rejections': rejections, 'yaw0': yaw0}
        set_robot_state(sim, cand.q)
        sim.command(np.asarray(cand.q, dtype=float))
        set_gripper(sim, 0.035, 20.0, 0.05)
        for _ in range(20):
            sim.step()
        start_q = sim.q().copy()
        start_ee, _ = sim.pose()

        expert = MonitoredExpert(sim, block)
        result = expert.skill_b()
        final_state = expert._obj()
        final_yaw = expert._yaw()
        lift = expert.max_obj_z - float(block_pos[2])
        moved_x = float(final_state['position'][0] - block_pos[0])
        record = {
            'mode': mode, 'seed': seed, 'sampled': True,
            'yaw0_deg': round(math.degrees(yaw0), 1),
            'yaw_deg': round(float(info['block_yaw']), 2),
            'block_start': block_pos.tolist(),
            'start_q': start_q.tolist(),
            'start_ee': start_ee.tolist(),
            'home_distance_rad': round(float(np.linalg.norm(start_q - np.asarray(HOME))), 4),
            'min_limit_margin': checks['min_limit_margin'],
            'block_clearance_m': checks['block_clearance_m'],
            'camera': checks['camera'],
            'success': bool(result['success']), 'reason': result['reason'],
            'metrics': result.get('metrics', {}),
            'final': result.get('final'),
            'lift_m': round(float(lift), 4),
            'lifted_flag': bool(lift > 0.04),
            'pushed_x_m': round(moved_x, 4),
            'max_joint_step_rad': round(expert.max_joint_step, 5),
            'min_home_distance_after_start': round(expert.home_min, 4),
            'ticks': len(expert.obj_trace),
            'rejections': rejections,
            'trace': result['trace'],
            'wall_seconds': round(time.time() - started, 2),
        }
        if preview:
            shot_dir = out / 'preview'
            shot_dir.mkdir(parents=True, exist_ok=True)
            shots = camera.capture()
            for name, arr in shots.items():
                from PIL import Image
                Image.fromarray(arr).save(shot_dir / f'{mode}_{seed}_{name}_start.png')
            shots = camera.capture()
            for name, arr in shots.items():
                from PIL import Image
                Image.fromarray(arr).save(shot_dir / f'{mode}_{seed}_{name}_end.png')
        return record
    finally:
        if p.isConnected(sim.client):
            p.disconnect(sim.client)


def summarize(records: list[dict]) -> dict:
    ok = [r for r in records if r.get('sampled')]
    def block(rows):
        k = sum(1 for r in rows if r['success'])
        return {'n': len(rows), 'success': k,
                'rate': round(k / len(rows), 4) if rows else 0.0,
                'ci95': wilson(k, len(rows))}
    by_mode = {m: block([r for r in ok if r['mode'] == m]) for m in MODES}
    by_yaw = {label: block([r for r in ok if (r['yaw0_deg'] == 0.0) == is_zero])
              for label, is_zero in (('long_Y_yaw0', True), ('long_X_yaw90', False))}
    lifted = [r for r in ok if r['lifted_flag']]
    steps = [r['max_joint_step_rad'] for r in ok]
    return {
        'episodes_requested': len(records),
        'sampled': len(ok),
        'success': sum(1 for r in ok if r['success']),
        'rate': round(sum(1 for r in ok if r['success']) / len(ok), 4) if ok else 0.0,
        'ci95': wilson(sum(1 for r in ok if r['success']), len(ok)),
        'by_mode': by_mode,
        'by_yaw_bin': by_yaw,
        'lifted_count': len(lifted),
        'lifted_episodes': [{'mode': r['mode'], 'seed': r['seed'],
                             'lift_m': r['lift_m']} for r in lifted],
        'max_joint_step_overall': max(steps) if steps else None,
        'mean_wall_seconds': round(float(np.mean([r['wall_seconds'] for r in ok])), 2) if ok else None,
        'failure_modes': {mode: sum(1 for r in ok if r['reason'] == mode)
                          for mode in sorted({r['reason'] for r in ok if not r['success']})},
        'rejections_total': {m: sum(int(r['rejections'].get(m, 0)) for r in ok + [x for x in records if not x.get('sampled')])
                             for m in sorted({k for r in records for k in r.get('rejections', {})})},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=36)
    ap.add_argument('--seed-base', type=int, default=50000000)
    ap.add_argument('--modes', nargs='+', default=list(MODES))
    ap.add_argument('--out', type=Path, default=ROOT / 'results/experiment_2_3/smoke')
    ap.add_argument('--preview-every', type=int, default=4,
                    help='每隔多少个 episode 保存一组起点/终态截图；0 表示不存')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    records = []
    for i in range(args.episodes):
        mode = args.modes[i % len(args.modes)]
        seed = args.seed_base + i
        rec = run_episode(mode, seed, args.out, preview=args.preview_every > 0
                          and i % args.preview_every == 0)
        records.append(rec)
        if rec.get('sampled'):
            print(f'[{i + 1}/{args.episodes}] {mode} seed={seed} yaw={rec["yaw0_deg"]} '
                  f'ok={rec["success"]} reason={rec["reason"]} lift={rec["lift_m"]:.3f} '
                  f'step={rec["max_joint_step_rad"]:.3f}', flush=True)
        else:
            print(f'[{i + 1}/{args.episodes}] {mode} seed={seed} NO_VALID_START '
                  f'{rec["rejections"]}', flush=True)

    with (args.out / 'episodes.jsonl').open('w', encoding='utf-8') as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + '\n')

    flat = [{k: v for k, v in rec.items()
             if k not in ('trace', 'rejections', 'camera', 'metrics', 'final')}
            for rec in records]
    keys = sorted({k for row in flat for k in row})
    with (args.out / 'episodes.csv').open('w', newline='', encoding='utf-8-sig') as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(flat)

    summary = summarize(records)
    (args.out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                           encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
