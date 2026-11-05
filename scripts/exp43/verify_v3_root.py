import json
from pathlib import Path

L = Path('/scratch/lijh/exp40/repo/results/experiment_4_3/data_expansion')
ROOT3 = Path('/scratch/lijh/exp43/export_v3')
v3 = json.loads((L / 'train_manifest_v3.json').read_text(encoding='utf-8'))
meta = [json.loads(l) for l in (ROOT3 / 'meta.jsonl').read_text(encoding='utf-8').splitlines()
        if l.strip()]
missing = []
for m in meta:
    for k in ('state_path', 'top_path', 'wrist_path'):
        p = ROOT3 / m[k]
        if not p.exists():
            missing.append(str(p))
        else:
            z = p.resolve()
            if not z.exists():
                missing.append(f'broken symlink {p} -> {z}')
print('meta rows', len(meta), 'missing files', len(missing))
if missing[:5]:
    print('examples', missing[:5])
train = [m for m in meta if m.get('split') == 'train']
print('train', len(train), 'A90', sum(1 for m in train if m['skill'] == 'A90'),
      'B90', sum(1 for m in train if m['skill'] == 'B90'))
print('val', sum(1 for m in meta if m.get('split') == 'val'),
      'test', sum(1 for m in meta if m.get('split') == 'test'))
# 与 v3 清单一致性
ids = {(e['skill'], int(e['episode_index'])) for e in v3['episodes']}
print('manifest v3 episodes', len(ids), 'meta train match',
      len({(m['skill'], int(m['episode_index'])) for m in train} & ids))
