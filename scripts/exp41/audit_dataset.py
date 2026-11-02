"""实验 4.1 数据审计：多样性（对象/机器人/联合终态）、分布统计、预注册 split manifest。

设计要点（prompt §5.5 / §8.4 / §10.4）：
  * **去重必须有容差**：物体终态按 (x,y,z,yaw) 欧氏距离 + 朝向差，机器人终态按 7 维 q 的 L∞ 距离；
  * **三种多样性分开报告**：Object Terminal / Robot Terminal / Joint Terminal；
  * **split 以场景/目标组为单位**，不允许把同一物理场景的近重复样本跨集合（这里按 goal 分组，
    A90 的 9 个目标 → train/val/test 各若干目标；B90 的 3 个目标区 + 2 种机器人起点分组）；
  * 统计只用于 4.2 决定 normalization，不在这里替换 checkpoint 的统计量。

用法：
    python scripts/exp41/audit_dataset.py --data data/exp41 --out results/experiment_4_1/dataset
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / 'src', ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

OBJ_POS_TOL = 0.005      # 5 mm
OBJ_YAW_TOL = math.radians(5.0)
Q_TOL = 0.02             # rad, L∞


def dedup_objects(entries: list[dict]) -> int:
    uniq = []
    for e in entries:
        p = np.asarray(e['obj_terminal_pos'], float)
        y = float(e['obj_terminal_yaw'])
        if not any(np.linalg.norm(p - u[0]) <= OBJ_POS_TOL and
                   abs(math.atan2(math.sin(y - u[1]), math.cos(y - u[1]))) <= OBJ_YAW_TOL
                   for u in uniq):
            uniq.append((p, y))
    return len(uniq)


def dedup_robots(entries: list[dict]) -> int:
    uniq = []
    for e in entries:
        q = np.asarray(e['robot_q_terminal'], float)
        if not any(np.max(np.abs(q - u)) <= Q_TOL for u in uniq):
            uniq.append(q)
    return len(uniq)


def dedup_joint(entries: list[dict]) -> int:
    uniq = []
    for e in entries:
        p = np.asarray(e['obj_terminal_pos'], float)
        y = float(e['obj_terminal_yaw'])
        q = np.asarray(e['robot_q_terminal'], float)
        if not any(np.linalg.norm(p - u[0]) <= OBJ_POS_TOL and
                   abs(math.atan2(math.sin(y - u[1]), math.cos(y - u[1]))) <= OBJ_YAW_TOL and
                   np.max(np.abs(q - u[2])) <= Q_TOL for u in uniq):
            uniq.append((p, y, q))
    return len(uniq)


def dataset_stats(data_root: Path, out: Path) -> dict:
    import pandas as pd

    stats = {}
    for skill in ('A90', 'B90'):
        root = data_root / skill
        if not (root / 'meta' / 'info.json').exists():
            continue
        eps = [json.loads(l) for l in (root / 'meta' / 'episodes.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
        S, A, T = [], [], []
        for ep in eps:
            idx = int(ep['episode_index'])
            df = pd.read_parquet(root / 'data' / f'chunk-{idx // 1000:03d}' / f'episode_{idx:06d}.parquet',
                                 columns=['observation.state', 'action', 'timestamp'])
            S.append(np.stack(df['observation.state'].to_numpy()).astype(np.float64))
            A.append(np.stack(df['action'].to_numpy()).astype(np.float64))
            T.append(df['timestamp'].to_numpy().astype(np.float64))
        S = np.concatenate(S); A = np.concatenate(A)
        # 训练用动作 = 相对当前 state 的关节增量 + 绝对归一化夹爪（与 pi05_droid 的语义对齐）
        dq = A[:, :7] - S[:, :7]
        grip = np.clip(S[:, 7:9].mean(axis=1) / 0.04, 0.0, 1.0)
        stats[skill] = {
            'n_episodes': len(eps), 'n_frames': int(S.shape[0]),
            'state': {'dim': int(S.shape[1]), 'min': S.min(0).round(5).tolist(),
                      'max': S.max(0).round(5).tolist(), 'mean': S.mean(0).round(5).tolist(),
                      'std': S.std(0).round(5).tolist(),
                      'q01': np.quantile(S, 0.01, axis=0).round(5).tolist(),
                      'q99': np.quantile(S, 0.99, axis=0).round(5).tolist()},
            'action_absolute': {'dim': int(A.shape[1]), 'min': A.min(0).round(5).tolist(),
                                'max': A.max(0).round(5).tolist(), 'mean': A.mean(0).round(5).tolist(),
                                'std': A.std(0).round(5).tolist()},
            'action_delta_7d': {'min': dq.min(0).round(5).tolist(), 'max': dq.max(0).round(5).tolist(),
                                'mean': dq.mean(0).round(5).tolist(), 'std': dq.std(0).round(5).tolist(),
                                'q01': np.quantile(dq, 0.01, axis=0).round(5).tolist(),
                                'q99': np.quantile(dq, 0.99, axis=0).round(5).tolist(),
                                'absmax': float(np.abs(dq).max()),
                                'frac_abs_gt_1_rad': float((np.abs(dq) > 1.0).mean()),
                                'note': 'π₀.₅(droid) 的动作是关节增量；此处给出可直接用于 4.2 的统计'},
            'gripper_normalized': {'min': float(grip.min()), 'max': float(grip.max()),
                                   'mean': float(grip.mean()),
                                   'frac_open': float((grip > 0.5).mean())},
            'fps_from_timestamps': float(1.0 / np.median(np.diff(np.concatenate(T)))),
        }
    (out / 'dataset_statistics.json').write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/exp41')
    ap.add_argument('--out', default='results/experiment_4_1/dataset')
    args = ap.parse_args(argv)
    data = Path(args.data) if Path(args.data).is_absolute() else ROOT / args.data
    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    # ---- 收集每个成功 episode 的终态（来自 collection records）----
    recs = {}
    rec_dir = out / 'records'
    for f in sorted(rec_dir.glob('*.json')) if rec_dir.exists() else []:
        r = json.loads(f.read_text(encoding='utf-8'))
        if r.get('strict_success'):
            recs[r['attempt_id']] = r
    by_skill = {'A90': [], 'B90': []}
    for r in recs.values():
        m = r['metrics']
        by_skill[r['skill']].append({
            'attempt_id': r['attempt_id'], 'goal_id': r['goal_id'],
            'config_id': r.get('config_id'), 'seed': r['seed'],
            'obj_terminal_pos': m['obj_terminal_pos'], 'obj_terminal_yaw': m['obj_terminal_yaw'],
            'robot_q_terminal': m['robot_q_terminal'],
            'metrics': {k: m.get(k) for k in ('position_error_m', 'yaw_error_deg', 'advance_x_m',
                                              'terminal_gap_m', 'wall_contact_steps',
                                              'wall_contact_force_max_n', 'final_y')},
        })

    audit = {'tolerances': {'object_pos_m': OBJ_POS_TOL, 'object_yaw_rad': OBJ_YAW_TOL,
                            'robot_q_l_inf_rad': Q_TOL},
             'by_skill': {}}
    for skill, entries in by_skill.items():
        if not entries:
            continue
        goals = {}
        for e in entries:
            goals.setdefault(e['goal_id'], []).append(e)
        audit['by_skill'][skill] = {
            'n_success': len(entries),
            'n_goals': len(goals),
            'n_per_goal': {g: len(v) for g, v in sorted(goals.items())},
            'object_terminal_uniques': dedup_objects(entries),
            'robot_terminal_uniques': dedup_robots(entries),
            'joint_terminal_uniques': dedup_joint(entries),
            'per_goal_object_uniques': {g: dedup_objects(v) for g, v in sorted(goals.items())},
        }
    (out / 'diversity_audit.json').write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, default=float), encoding='utf-8')

    # ---- 预注册 split（按目标组，不按帧）----
    a_goals = sorted({e['goal_id'] for e in by_skill['A90']})
    b_configs = sorted({e.get('config_id') for e in by_skill['B90'] if e.get('config_id')})
    split = {
        'policy': 'split by goal/config group, never by frame; no near-duplicate scene may straddle splits',
        'A90': {'train_goals': a_goals[:6], 'val_goals': a_goals[6:8], 'test_goals': a_goals[8:],
                'heldout_role': 'test goals are never seen during training'},
        'B90': {'train_configs': b_configs[:4], 'val_configs': b_configs[4:5],
                'test_configs': b_configs[5:],
                'heldout_role': 'test configs hold out a (object start, goal, robot start) combination'},
        'composition_conditions_future': {
            'note': '实验 4.1 不产生任何 A→B 组合轨迹；4.2+ 将用 A 终态 × B 起点配对，'
                    '组合配对本身在训练中从未出现（held-out composition）',
            'a_terminal_states': 'results/experiment_4_1/dataset/diversity_audit.json + records/*.json',
            'b_start_states': 'results/experiment_4_1/scene_space/b_start_states.json',
        },
        'generated_from': 'collection records (strict_success only)',
    }
    (out / 'split_manifest.json').write_text(
        json.dumps(split, ensure_ascii=False, indent=2), encoding='utf-8')

    stats = dataset_stats(data, out) if (data / 'A90' / 'meta').exists() else {}
    print(json.dumps({'diversity': audit['by_skill'], 'split': split,
                      'stats_keys': list(stats)}, ensure_ascii=False, indent=2, default=float)[:3000])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
