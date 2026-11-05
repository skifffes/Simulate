"""实验 4.3 训练集清单 v2：**只取四个无重复目录**，分配全局唯一 ID，并给实际选中的
parquet/MP4 记录 SHA256。

修复两个问题：
  1. 去重：只读 `data_a90a`、`data_a90b`、`data_b90a`、`data_b90b`（A90 140 + B90 113），
     不读首批 `data/`（那 9 条是 offset 碰撞的重采）；并按 (skill, seed) 二次去重兜底。
  2. 哈希：以**相对 /scratch/lijh/exp43 的完整路径**为键，对每条选中 episode 的 parquet 与两个
     相机 MP4 计算 SHA256（不再被同名叶目录覆盖）。

输出：
  results/experiment_4_3/data_expansion/train_manifest_v2.json
  results/experiment_4_3/data_expansion/selected_artifacts_sha256.json
  results/experiment_4_3/data_expansion/coverage_v2.json
"""
from __future__ import annotations

import glob
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

SRV = Path('/scratch/lijh/exp43')
OUT = Path('results/experiment_4_3/data_expansion')
SOURCES = [('a90a', 'A90'), ('a90b', 'A90'), ('b90a', 'B90'), ('b90b', 'B90')]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    seen = set()
    episodes, artifacts = [], {}
    for tag, skill in SOURCES:
        ddir = SRV / f'data_{tag}'
        rec_dir = SRV / tag / 'records'
        pq = {int(Path(p).stem.split('_')[-1]): Path(p)
              for p in glob.glob(str(ddir / skill / 'data' / '**' / '*.parquet'), recursive=True)}
        vids = defaultdict(list)
        for p in glob.glob(str(ddir / skill / 'videos' / '**' / '*.mp4'), recursive=True):
            name = Path(p).stem
            try:
                idx = int(name.split('_')[-1])
            except ValueError:
                continue
            vids[idx].append(Path(p))
        for f in sorted(rec_dir.glob('*.json')):
            rec = json.loads(f.read_text(encoding='utf-8'))
            if not rec.get('strict_success') or rec.get('stored_episode_index') is None:
                continue
            key = (skill, rec['seed'])
            if key in seen:
                continue
            seen.add(key)
            idx = int(rec['stored_episode_index'])
            p = pq.get(idx)
            if p is None:
                continue
            gid = len(episodes) + 1
            rel_p = str(p.relative_to(SRV))
            rel_v = [str(v.relative_to(SRV)) for v in sorted(vids.get(idx, []))]
            episodes.append({
                'global_episode_id': gid, 'skill': skill, 'source_tag': tag,
                'source_episode_index': idx, 'seed': rec['seed'],
                'config_id': rec.get('config_id'), 'goal_id': rec.get('goal_id'),
                'n_control_steps': rec.get('n_control_steps'),
                'parquet': rel_p, 'videos': rel_v,
                'metrics': {k: (rec.get('metrics') or {}).get(k) for k in
                            ('advance_x_m', 'wall_contact_steps', 'terminal_gap_m', 'tilt_deg')},
            })
            artifacts[rel_p] = sha256(p)
            for rv in rel_v:
                artifacts[rv] = sha256(SRV / rv)

    n_parq = sum(1 for k in artifacts if k.endswith('.parquet'))
    n_mp4 = sum(1 for k in artifacts if k.endswith('.mp4'))
    by_cfg = Counter(f"{e['skill']}:{e['config_id']}" for e in episodes)
    manifest = {
        'version': 'exp43-train-manifest-v2',
        'sources': [{'tag': t, 'skill': s, 'dir': f'data_{t}'} for t, s in SOURCES],
        'excluded': {'dir': 'data', 'reason': '首批 9 条与 a90a/b90a 的 (config, seed) 重复（offset 碰撞）'},
        'n_episodes': len(episodes),
        'n_by_skill': dict(Counter(e['skill'] for e in episodes)),
        'n_by_config': dict(sorted(by_cfg.items())),
        'unique_(skill,seed)': len({(e['skill'], e['seed']) for e in episodes}),
        'episodes': episodes,
    }
    (OUT / 'train_manifest_v2.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'selected_artifacts_sha256.json').write_text(
        json.dumps({'root': str(SRV), 'n_parquet': n_parq, 'n_mp4': n_mp4,
                    'files': artifacts}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: manifest[k] for k in ('n_episodes', 'n_by_skill', 'unique_(skill,seed)')},
                     ensure_ascii=False))
    print(json.dumps(manifest['n_by_config'], ensure_ascii=False, indent=1))
    print('hashed', n_parq, 'parquet', n_mp4, 'mp4')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
