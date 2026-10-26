"""实验 2.3 阶段 0：审计现有原子数据、起始状态覆盖、A 终态与评测协议。

（只读脚本。）

只读取已有产物（实验 2 评测 JSON/CSV、实验 2.1 审计与快照），不重新训练、
不跑仿真、不覆盖旧结果。输出写到 results/experiment_2_3/audit/。

用法：
    python scripts/audit_experiment_2_3.py
"""
from __future__ import annotations

import csv
import json
import math
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'experiment_2_3' / 'audit'
SNAPSHOTS = ROOT / 'results' / 'experiment_2_1' / 'snapshots'
EXP2 = ROOT / 'results' / 'experiment_2'


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% 置信区间。"""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate_block(successes: int, n: int) -> dict:
    lo, hi = wilson(successes, n)
    return {'success': successes, 'n': n, 'rate': round(successes / n, 4) if n else 0.0,
            'ci95': [round(lo, 4), round(hi, 4)]}


def load_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def b_by_yaw(path: Path) -> dict:
    """按物体初始朝向把 B 的逐集结果分组（长轴 Y≈0° / 长轴 X≈90°）。"""
    data = load_json(path)
    groups = {'long_Y_yaw0': [], 'long_X_yaw90': []}
    for ep in data['episodes']:
        yaw = float(ep['scene']['yaw'])
        folded = abs(math.atan2(math.sin(yaw), math.cos(yaw)))
        key = 'long_Y_yaw0' if folded < math.pi / 4 else 'long_X_yaw90'
        groups[key].append(ep)
    out = {'n_total': len(data['episodes']), 'summary': data['summary']}
    for key, eps in groups.items():
        block = rate_block(sum(1 for e in eps if e['success']), len(eps))
        block['yaw_deg'] = [round(math.degrees(float(e['scene']['yaw'])), 1) for e in eps]
        block['failure_modes'] = {}
        for e in eps:
            if not e['success']:
                mode = e.get('failure') or 'unknown'
                block['failure_modes'][mode] = block['failure_modes'].get(mode, 0) + 1
        out[key] = block
    return out


def terminal_stats() -> dict:
    """A0/A90 真实终态（实验 2.1 的 40 个冻结 VLA rollout 快照）。"""
    home = [0.0, -0.5, 0.0, -2.1, 0.0, 1.65, 0.78]
    out = {}
    for skill in ('A0', 'A90'):
        rows = []
        for path in sorted(SNAPSHOTS.glob(f'{skill}_*.json')):
            if path.name in ('collection_log.json', 'terminal_summary.json'):
                continue
            snap = load_json(path)
            rows.append(snap)
        if not rows:
            continue
        qs = [r['robot']['q'] for r in rows]
        ees = [r['robot']['ee_position'] for r in rows]
        dist_home = [math.dist(q, home) for q in qs]
        yaws = [math.degrees(r['object']['yaw_rad']) for r in rows]
        out[skill] = {
            'n': len(rows),
            'q_mean': [round(statistics.fmean(q[i] for q in qs), 4) for i in range(7)],
            'q_std': [round(statistics.pstdev(q[i] for q in qs), 4) for i in range(7)],
            'q_min': [round(min(q[i] for q in qs), 4) for i in range(7)],
            'q_max': [round(max(q[i] for q in qs), 4) for i in range(7)],
            'dist_to_home_rad': {
                'mean': round(statistics.fmean(dist_home), 3),
                'min': round(min(dist_home), 3),
                'max': round(max(dist_home), 3),
            },
            'ee_position_mean': [round(statistics.fmean(e[i] for e in ees), 4) for i in range(3)],
            'ee_position_min': [round(min(e[i] for e in ees), 4) for i in range(3)],
            'ee_position_max': [round(max(e[i] for e in ees), 4) for i in range(3)],
            'object_yaw_deg_mean': round(statistics.fmean(yaws), 2),
            'object_yaw_deg_min': round(min(yaws), 2),
            'object_yaw_deg_max': round(max(yaws), 2),
            'joint7_mean': round(statistics.fmean(q[6] for q in qs), 4),
            'ticks_mean': round(statistics.fmean(r['ticks'] for r in rows), 1),
        }
    return out


def git_commit() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                       cwd=ROOT, text=True).strip()
    except Exception:
        return 'unknown'


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    audit21 = load_json(ROOT / 'results' / 'experiment_2_1' / 'audit' / 'summary.json')
    coverage21 = load_json(ROOT / 'results' / 'experiment_2_1' / 'coverage' / 'coverage.json')

    start_states = {
        skill: {
            'unique_rows': audit21['start_joint_stats'][skill]['unique_rows'],
            'max_abs_dev_from_home': audit21['start_joint_stats'][skill]['max_abs_dev_from_home'],
            'grip0_mean': audit21['start_joint_stats'][skill]['grip0_mean'],
        }
        for skill in ('A0', 'A90', 'B', 'C')
    }

    b_vla = b_by_yaw(EXP2 / 'eval_atomic' / 'eval_B.json')
    b_expert = b_by_yaw(EXP2 / 'expert_atomic' / 'eval_B.json')

    terminals = terminal_stats()

    composition = {}
    with (EXP2 / 'eval_composition' / 'episodes.csv').open(encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            chain = row['chain']
            composition.setdefault(chain, {'n': 0, 'chain_success': 0,
                                           'A_success': 0, 'B_success': 0})
            composition[chain]['n'] += 1
            composition[chain]['chain_success'] += row['chain_success'] == 'True'
            composition[chain]['A_success'] += row['A_success'] == 'True'
            composition[chain]['B_success'] += row['B_success'] == 'True'

    audit = {
        'repo_commit': git_commit(),
        'dataset': audit21['dataset'],
        'start_states': start_states,
        'object_yaw_coverage': {
            skill: {
                'attempts': audit21['object_distributions'][skill]['attempts'],
                'success': audit21['object_distributions'][skill]['success'],
                'x_range': audit21['object_distributions'][skill]['x_range'],
                'y_range': audit21['object_distributions'][skill]['y_range'],
                'n_yaw_values': len(audit21['object_distributions'][skill]['yaw_values']),
                'yaw_min_deg': round(math.degrees(min(audit21['object_distributions'][skill]['yaw_values'])), 1),
                'yaw_max_deg': round(math.degrees(max(audit21['object_distributions'][skill]['yaw_values'])), 1),
            }
            for skill in ('B', 'C')
        },
        'b_atomic_by_yaw': {'vla': b_vla, 'expert': b_expert},
        'a_terminals': terminals,
        'coverage_state_distance': {
            'starts_unique': coverage21['starts_unique'],
            'terminal_to_home_rad_mean': {
                k: round(statistics.fmean(v), 3) for k, v in coverage21['terminal_to_home_rad'].items()
            },
            'terminal_to_demo_knn_rad_mean': {
                k: round(statistics.fmean(v), 3) for k, v in coverage21['terminal_to_demo_knn_rad'].items()
            },
            'terminal_to_B_knn_rad_mean': {
                k: round(statistics.fmean(v), 3) for k, v in coverage21['terminal_to_B_knn_rad'].items()
            },
            'terminal_to_C_knn_rad_mean': {
                k: round(statistics.fmean(v), 3) for k, v in coverage21['terminal_to_C_knn_rad'].items()
            },
        },
        'composition_exp2': composition,
    }

    (OUT / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2),
                                    encoding='utf-8')

    # 便于人工核对的扁平 CSV
    rows = []
    for model in ('vla', 'expert'):
        data = b_vla if model == 'vla' else b_expert
        for group in ('long_Y_yaw0', 'long_X_yaw90'):
            block = data[group]
            rows.append({
                'model': model, 'group': group, 'success': block['success'],
                'n': block['n'], 'rate': block['rate'],
                'ci_low': block['ci95'][0], 'ci_high': block['ci95'][1],
                'failure_modes': json.dumps(block['failure_modes'], ensure_ascii=False),
            })
    with (OUT / 'b_by_yaw.csv').open('w', newline='', encoding='utf-8-sig') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({
        'start_states': start_states,
        'b_vla': {k: {kk: b_vla[k][kk] for kk in ('success', 'n', 'rate', 'ci95')}
                  for k in ('long_Y_yaw0', 'long_X_yaw90')},
        'b_expert': {k: {kk: b_expert[k][kk] for kk in ('success', 'n', 'rate', 'ci95')}
                     for k in ('long_Y_yaw0', 'long_X_yaw90')},
        'a_terminals': {k: {'n': v['n'], 'dist_to_home_rad': v['dist_to_home_rad'],
                            'ee_position_mean': v['ee_position_mean']}
                        for k, v in terminals.items()},
        'composition_exp2': composition,
    }, ensure_ascii=False, indent=2))
    print(f'-> {OUT}')


if __name__ == '__main__':
    main()
