"""实验 4.3：编译新增采集结果（多并行目录）。

输出到 results/experiment_4_3/data_expansion/：
  strict_success_results.csv / dataset_manifest.json / coverage_comparison.json /
  train_split_manifest.json / large_artifacts_sha256.json
"""
from __future__ import annotations

import csv
import glob
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

SRV = Path('/scratch/lijh/exp43')
OUT = Path('results/experiment_4_3/data_expansion')

TRAIN_A = ['A_x-3_y-3', 'A_x00_y-3', 'A_x+3_y-3', 'A_x-3_y00', 'A_x00_y00', 'A_x+3_y00']
VAL_A = ['A_x-3_y+3', 'A_x00_y+3']
TEST_A = ['A_x+3_y+3']
TRAIN_B = ['B_wall_y_low|B_obj_y015|B_robot_home', 'B_wall_y_low|B_obj_y015|B_robot_offset',
           'B_wall_y_mid|B_obj_y021|B_robot_home', 'B_wall_y_mid|B_obj_y021|B_robot_offset']
VAL_B = ['B_wall_y_high|B_obj_y027|B_robot_home']
TEST_B = ['B_wall_y_high|B_obj_y027|B_robot_offset']


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def data_dir_of(tag: str) -> Path:
    p = SRV / f'data_{tag}'
    return p if p.exists() else SRV / 'data'


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tags = sorted({p.name for p in SRV.glob('a90*') if (p / 'records').is_dir()} |
                  {p.name for p in SRV.glob('b90*') if (p / 'records').is_dir()})

    rows, episodes = [], []
    for tag in tags:
        skill = 'A90' if tag.startswith('a90') else 'B90'
        ddir = data_dir_of(tag)
        for f in sorted((SRV / tag / 'records').glob('*.json')):
            rec = json.loads(f.read_text(encoding='utf-8'))
            rows.append({'tag': tag, 'skill': skill, 'config_id': rec.get('config_id'),
                         'seed': rec.get('seed'), 'strict_success': bool(rec.get('strict_success')),
                         'failure_reason': rec.get('failure_reason') or '',
                         'n_control_steps': rec.get('n_control_steps'),
                         'attempt_id': rec.get('attempt_id'),
                         'stored_episode_index': rec.get('stored_episode_index')})
            if rec.get('strict_success') and rec.get('stored_episode_index') is not None:
                episodes.append({'tag': tag, 'skill': skill,
                                 'episode_index': rec['stored_episode_index'],
                                 'seed': rec['seed'], 'config_id': rec.get('config_id'),
                                 'goal_id': rec.get('goal_id'),
                                 'n_control_steps': rec.get('n_control_steps'),
                                 'metrics': {k: (rec.get('metrics') or {}).get(k) for k in
                                             ('advance_x_m', 'wall_contact_steps',
                                              'terminal_gap_m', 'tilt_deg')},
                                 'data_dir': str(ddir / skill)})

    with open(OUT / 'strict_success_results.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ['tag'])
        w.writeheader()
        w.writerows(rows)

    # 数据落盘情况（info.json）
    dirs = {}
    for tag in tags:
        ddir = data_dir_of(tag)
        for skill in ('A90', 'B90'):
            info = ddir / skill / 'meta' / 'info.json'
            if info.exists():
                j = json.loads(info.read_text(encoding='utf-8'))
                dirs[f'{ddir.name}/{skill}'] = {
                    'total_episodes': j.get('total_episodes'),
                    'total_frames': j.get('total_frames'),
                    'info_sha256': sha256(info),
                }

    by_cfg = defaultdict(lambda: {'attempts': 0, 'stored': 0})
    for r in rows:
        key = f"{r['skill']}:{r['config_id']}"
        by_cfg[key]['attempts'] += 1
        by_cfg[key]['stored'] += int(r['strict_success'] and r['stored_episode_index'] is not None)

    new_seeds = {r['seed'] for r in rows if r['strict_success']}
    old_seeds = set()
    for f in glob.glob('results/experiment_4_1/dataset/records/*.json'):
        rec = json.loads(open(f, encoding='utf-8').read())
        if rec.get('seed') is not None:
            old_seeds.add(rec['seed'])

    used_a = {e['config_id'] for e in episodes if e['skill'] == 'A90'}
    used_b = {e['config_id'] for e in episodes if e['skill'] == 'B90'}

    coverage = {
        'new_episodes_by_skill': dict(Counter(e['skill'] for e in episodes)),
        'new_attempts_by_skill': dict(Counter(r['skill'] for r in rows)),
        'new_success_by_skill': {s: sum(1 for e in episodes if e['skill'] == s)
                                  for s in ('A90', 'B90')},
        'by_config': {k: dict(v) for k, v in sorted(by_cfg.items())},
        'failure_reasons': dict(Counter(r['failure_reason'] for r in rows if not r['strict_success'])),
        'data_dirs': dirs,
        'old_train_counts': {'A90': 30, 'B90': 31},
        'combined_train_counts': {
            'A90': 30 + sum(1 for e in episodes if e['skill'] == 'A90'),
            'B90': 31 + sum(1 for e in episodes if e['skill'] == 'B90'),
        },
    }
    split = {
        'A90': {'train_configs': TRAIN_A, 'used_by_new': sorted(used_a),
                'val_configs': VAL_A, 'test_configs': TEST_A,
                'uses_val_or_test': sorted(used_a & set(VAL_A + TEST_A))},
        'B90': {'train_configs': TRAIN_B, 'used_by_new': sorted(used_b),
                'val_configs': VAL_B, 'test_configs': TEST_B,
                'uses_val_or_test': sorted(used_b & set(VAL_B + TEST_B))},
        'seed_check': {
            'n_success_episodes': len(episodes),
            'n_unique_new_seeds': len(new_seeds),
            'n_old_seeds': len(old_seeds),
            'seed_overlap_with_old': len(new_seeds & old_seeds),
            'new_seed_ranges': {'A90': 'offset 40-67', 'B90': 'offset 40-79'},
        },
        'duplicate_episode_check': {
            'unique_(skill,seed)': len({(e['skill'], e['seed']) for e in episodes}),
            'n_success_episodes': len(episodes),
        },
    }

    (OUT / 'dataset_manifest.json').write_text(
        json.dumps({'episodes': episodes, 'data_dirs': dirs}, ensure_ascii=False, indent=2),
        encoding='utf-8')
    (OUT / 'coverage_comparison.json').write_text(
        json.dumps(coverage, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'train_split_manifest.json').write_text(
        json.dumps(split, ensure_ascii=False, indent=2), encoding='utf-8')

    arts = {}
    for tag in tags:
        ddir = data_dir_of(tag)
        for skill in ('A90', 'B90'):
            d = ddir / skill
            if d.exists():
                n_parq = len(list(d.glob('data/**/*.parquet')))
                n_mp4 = len(list(d.glob('videos/**/*.mp4')))
                arts[f'{d.name}'] = {'parquet': n_parq, 'mp4': n_mp4,
                                      'bytes': sum(p.stat().st_size for p in d.rglob('*')
                                                   if p.is_file())}
    (OUT / 'large_artifacts_sha256.json').write_text(
        json.dumps({'root': str(SRV), 'per_dataset': arts,
                    'note': '大工件留在服务器；此处记录文件数与字节数，info.json 的 sha256 见 coverage'},
                   ensure_ascii=False, indent=2), encoding='utf-8')

    print('new success by skill:', coverage['new_success_by_skill'])
    print('data dirs:', json.dumps(dirs, ensure_ascii=False))
    print('seed overlap with old:', split['seed_check']['seed_overlap_with_old'])
    print('uses val/test:', split['A90']['uses_val_or_test'], split['B90']['uses_val_or_test'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
