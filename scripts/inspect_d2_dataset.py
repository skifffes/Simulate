import json
from pathlib import Path
import pyarrow.parquet as pq

root = Path('data/lerobot_panda_e23_D2')
info = json.loads((root / 'meta/info.json').read_text())
print('info keys', info.keys())
print('total_episodes', info.get('total_episodes'), 'total_frames', info.get('total_frames'), 'fps', info.get('fps'))
print('features', json.dumps(info.get('features', {}), indent=1)[:1500])
tasks = [json.loads(x) for x in (root / 'meta/tasks.jsonl').read_text().splitlines() if x.strip()]
print('tasks', tasks)
eps = [json.loads(x) for x in (root / 'meta/episodes.jsonl').read_text().splitlines() if x.strip()]
print('episodes n', len(eps), 'keys', eps[0].keys())
for e in eps[:5]:
    print('ep', {k: e[k] for k in e if k in ('episode_index', 'tasks', 'length')})
files = sorted(root.glob('data/chunk-*/*.parquet'))
print('parquet files', len(files), files[0])
pf = pq.ParquetFile(files[0])
print('columns', pf.schema_arrow.names)
print('rows', pf.metadata.num_rows)
t = pf.read_row_group(0, columns=[c for c in pf.schema_arrow.names if c in ('observation.state','action','task','episode_index','frame_index','timestamp','index')])
print(t.slice(0, 2).to_pylist())
