"""实验 4.1 计划生成 + Pilot + 有界采集。

模式：
  --mode pilot    只跑尝试（默认不录帧），产出 attempts.csv 与成功率统计
  --mode collect  只把**严格评价通过**的尝试写入 LeRobot（录帧），并写质量/多样性审计

计划是预注册的（goal × start × seed 全组合，种子固定），不允许事后挑选容易的目标。

用法：
    python scripts/exp41/collect41.py --mode pilot --out results/experiment_4_1/pilot
    python scripts/exp41/collect41.py --mode collect --target-a 40 --target-b 40 \
        --out results/experiment_4_1/dataset --data data/exp41
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / 'src', ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from exp41 import scene41  # noqa: E402
from exp41.run41 import (A90_PROMPT_TEMPLATE, ACTION_CONTRACT_VERSION, PHYSICS_HZ, POLICY_HZ,
                         STEPS_PER_ACTION, B90_PROMPT_TEMPLATE, run_attempt)  # noqa: E402
from exp41.strict_eval import DEFAULT_CONFIG, asdict as _asdict  # noqa: E402

SEED_BASE = 410000


def _goal(kind: str, goal_id: str):
    pool = scene41.A_GOALS if kind == 'A90' else scene41.B_GOALS
    for g in pool:
        if g.goal_id == goal_id:
            return g
    raise KeyError(goal_id)


def build_plan(skill: str, *, per_condition: int = 2) -> list[dict]:
    """预注册计划：**按轮次交错**覆盖所有条件（先保证目标/条件的覆盖面，再加密种子）。

    顺序很关键：如果按"每个条件连续跑满种子"，上限一到就只覆盖前几个目标，
    数据集会出现"45 条但只有 4 个目标"的偏置。
    """
    plan = []
    if skill == 'A90':
        conds = [{'goal_id': g.goal_id, 'obj_start_id': None, 'robot_start_id': None,
                  'config_id': g.goal_id} for g in scene41.A_GOALS]
        for k in range(per_condition):
            for gi, c in enumerate(conds):
                plan.append({'skill': 'A90', **c, 'seed': SEED_BASE + gi * 100 + k})
    else:
        conds = []
        for gi, (obj_id, goal_id) in enumerate(scene41.B_ALIGNED_PAIRS):
            for robot in scene41.B_ROBOT_STARTS:
                conds.append({'goal_id': goal_id, 'obj_start_id': obj_id,
                              'robot_start_id': robot['robot_start_id'],
                              'config_id': f'{goal_id}|{obj_id}|{robot["robot_start_id"]}'})
        for k in range(per_condition):
            for gi, c in enumerate(conds):
                plan.append({'skill': 'B90', **c, 'seed': SEED_BASE + 10000 + gi * 100 + k})
    return plan


def run_one(task: dict, *, record_frames: bool, camera_version: str = 'v3'):
    goal = _goal(task['skill'], task['goal_id'])
    obj = None
    robot = None
    if task['skill'] == 'B90':
        obj = next(o for o in scene41.B_OBJECT_STARTS if o['start_id'] == task['obj_start_id'])
        robot = next(r for r in scene41.B_ROBOT_STARTS if r['robot_start_id'] == task['robot_start_id'])
    return run_attempt(task['skill'], seed=task['seed'], goal=goal, obj_start=obj,
                       robot_start=robot, record_frames=record_frames,
                       camera_version=camera_version)


def _row(rec: dict) -> dict:
    m = rec.get('metrics', {})
    return {
        'attempt_id': rec['attempt_id'], 'skill': rec['skill'], 'goal_id': rec['goal_id'],
        'seed': rec['seed'], 'config_id': rec.get('config_id'),
        'strict_success': rec['strict_success'], 'failure_reason': rec['failure_reason'] or '',
        'expert_ok': rec['expert_ok'], 'expert_reason': rec['expert_reason'],
        'n_control_steps': rec['n_control_steps'],
        'position_error_m': m.get('position_error_m', ''),
        'yaw_error_deg': m.get('yaw_error_deg', ''),
        'lift_m': m.get('lift_m', ''),
        'transport_m': m.get('transport_m', ''),
        'advance_x_m': m.get('advance_x_m', ''),
        'terminal_gap_m': m.get('terminal_gap_m', ''),
        'wall_contact_steps': m.get('wall_contact_steps', ''),
        'wall_contact_force_max_n': m.get('wall_contact_force_max_n', ''),
        'final_y': m.get('final_y', ''),
        'tilt_deg': m.get('tilt_deg', ''),
        'final_speed_m_s': m.get('final_speed_m_s', ''),
        'obj_terminal_pos': json.dumps(m.get('obj_terminal_pos', [])),
        'robot_q_terminal': json.dumps(m.get('robot_q_terminal', [])),
        'wall_seconds': rec.get('wall_seconds'),
    }


def pilot(args) -> int:
    out = Path(args.out)
    (out / 'A90').mkdir(parents=True, exist_ok=True)
    (out / 'B90').mkdir(parents=True, exist_ok=True)
    all_rows = []
    summaries = {}
    for skill in ('A90', 'B90'):
        plan = build_plan(skill, per_condition=args.per_condition)
        rows = []
        t0 = time.time()
        for i, task in enumerate(plan, 1):
            att = run_one(task, record_frames=False, camera_version=args.camera_version)
            rec = dict(att.record)
            rec['config_id'] = task['config_id']
            row = _row(rec)
            rows.append(row)
            (out / skill / 'records').mkdir(exist_ok=True)
            (out / skill / 'records' / f'{rec["attempt_id"]}.json').write_text(
                json.dumps(rec, ensure_ascii=False, indent=2, default=float), encoding='utf-8')
            print(f'[{skill} {i}/{len(plan)}] {task["config_id"]} seed={task["seed"]} '
                  f'success={row["strict_success"]} reason={row["failure_reason"]} '
                  f'{row["wall_seconds"]}s', flush=True)
        all_rows += rows
        by_cfg = {}
        for r in rows:
            d = by_cfg.setdefault(r['config_id'], [0, 0])
            d[1] += 1
            d[0] += int(r['strict_success'])
        summaries[skill] = {
            'n_attempts': len(rows), 'n_success': sum(1 for r in rows if r['strict_success']),
            'success_rate': sum(1 for r in rows if r['strict_success']) / max(1, len(rows)),
            'by_config': by_cfg,
            'failure_reasons': {k: sum(1 for r in rows if r['failure_reason'] == k)
                                for k in {r['failure_reason'] for r in rows}},
            'elapsed_s': round(time.time() - t0, 1),
            'object_terminal_uniques': len({tuple(np.round(json.loads(r['obj_terminal_pos']), 4))
                                            for r in rows if r['strict_success']}),
            'robot_q_terminal_uniques': len({tuple(np.round(json.loads(r['robot_q_terminal']), 3))
                                             for r in rows if r['strict_success']}),
        }
    keys = list(all_rows[0].keys())
    with open(out / 'attempts.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(all_rows)
    (out / 'pilot_summary.json').write_text(json.dumps({
        'plan': {'A90': len(build_plan('A90', per_condition=args.per_condition)),
                 'B90': len(build_plan('B90', per_condition=args.per_condition)),
                 'per_condition_seeds': args.per_condition, 'seed_base': SEED_BASE,
                 'b_aligned_pairs': scene41.B_ALIGNED_PAIRS},
        'camera_version': args.camera_version,
        'eval_config': asdict(DEFAULT_CONFIG),
        'summaries': summaries,
    }, ensure_ascii=False, indent=2, default=float), encoding='utf-8')
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


def collect(args) -> int:
    from vla.atomic_tasks.config2 import feature_dict  # noqa: E402
    from vla.lerobot_writer import LeRobotV21Writer  # noqa: E402

    data_root = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.fresh_records:
        # 只清理本次运行的尝试记录；数据集目录由 writer 自己 resume/finalize
        import shutil
        rec_dir = out / 'records'
        if rec_dir.exists():
            shutil.rmtree(rec_dir)
        print(f'[collect] cleared {rec_dir}')
    quality_rows = []
    manifests = {}
    for skill, target in (('A90', args.target_a), ('B90', args.target_b)):
        if target <= 0:
            continue
        shard = data_root / skill
        writer = LeRobotV21Writer(shard, feature_dict(), POLICY_HZ, robot_type='panda',
                                  task=None, resume=True)
        plan = build_plan(skill, per_condition=args.collect_seeds)
        max_attempts = min(len(plan), int(target * 3))
        stored = len(writer.episodes)
        attempts = 0
        t0 = time.time()
        for task in plan:
            if stored >= target or attempts >= max_attempts:
                break
            attempts += 1
            att = run_one(task, record_frames=True, camera_version=args.camera_version)
            rec = dict(att.record)
            rec['config_id'] = task['config_id']
            quality_rows.append(_row(rec))
            if rec['strict_success'] and att.actions is not None and len(att.actions) >= 10:
                prompt = rec['prompt']
                stored_info = writer.add_episode(att.frames, att.states, att.actions, prompt)
                rec['stored_episode_index'] = stored_info['episode_index']
                stored += 1
            (out / 'records').mkdir(exist_ok=True)
            (out / 'records' / f'{rec["attempt_id"]}.json').write_text(
                json.dumps(rec, ensure_ascii=False, indent=2, default=float), encoding='utf-8')
            print(f'[{skill} {stored}/{target} attempt {attempts}] {task["config_id"]} '
                  f'success={rec["strict_success"]} {rec["wall_seconds"]}s', flush=True)
        writer.finalize()
        manifests[skill] = {'dataset': str(shard), 'episodes': len(writer.episodes),
                            'frames': writer.total_frames, 'attempts': attempts,
                            'target': target, 'elapsed_s': round(time.time() - t0, 1)}
    with open(out / 'quality_audit.csv', 'w', newline='', encoding='utf-8') as fh:
        keys = list(quality_rows[0].keys()) if quality_rows else ['attempt_id']
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(quality_rows)
    (out / 'collection_summary.json').write_text(json.dumps({
        'manifests': manifests, 'camera_version': args.camera_version,
        'camera_sha256': scene41.camera_v2.CAMERA_CONFIG_SHA256,
        'action_contract_version': ACTION_CONTRACT_VERSION,
        'physics_hz': PHYSICS_HZ, 'policy_hz': POLICY_HZ, 'steps_per_action': STEPS_PER_ACTION,
        'eval_config': asdict(DEFAULT_CONFIG),
        'n_attempts': len(quality_rows),
        'n_success': sum(1 for r in quality_rows if r['strict_success']),
    }, ensure_ascii=False, indent=2, default=float), encoding='utf-8')
    print(json.dumps(manifests, ensure_ascii=False, indent=2))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['pilot', 'collect'], required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--data', default='data/exp41')
    ap.add_argument('--per-condition', type=int, default=2, help='pilot: 每个条件尝试的种子数')
    ap.add_argument('--collect-seeds', type=int, default=12, help='collect: 每个条件的种子池大小')
    ap.add_argument('--target-a', type=int, default=0)
    ap.add_argument('--target-b', type=int, default=0)
    ap.add_argument('--camera-version', default='v3')
    ap.add_argument('--fresh-records', action='store_true',
                    help='采集前清空 dataset/records，保证 record ↔ episode 的一一对应')
    args = ap.parse_args(argv)
    return pilot(args) if args.mode == 'pilot' else collect(args)


if __name__ == '__main__':
    raise SystemExit(main())
