import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import av
from PIL import Image

ROOT = Path('/scratch/lijh/simulate_vla')
OUT = ROOT / 'results/experiment_2_5/d2_c3_coverage/matched_frames'
OUT.mkdir(parents=True, exist_ok=True)
shard = ROOT / 'data/_shards_e23/B_D2'
ep = 66
st = pq.read_table(shard / f'data/chunk-000/episode_{ep:06d}.parquet',
                   columns=['observation.state']).to_pydict()['observation.state']
q0 = np.array(st[0][:7])
rows = [json.loads(x) for x in (ROOT / 'results/experiment_2_3/data/D2_episodes.jsonl').read_text().splitlines() if x.strip()]
ok = [r for r in rows if r['success'] and r['reason'] == 'ok']
j = int(np.argmin([np.linalg.norm(q0 - np.array(r['start_q'])) for r in ok]))
r = ok[j]
print('ep66 collection', r['index'], r['mode'], r['spec']['group'], r['scene']['block_start'],
      r['scene']['block_yaw'], 'len', len(st))


def grab(video, idx, out):
    c = av.open(str(video))
    for i, f in enumerate(c.decode(video=0)):
        if i == idx:
            f.to_image().save(out)
            print('saved', out, f.width, f.height)
            break
    c.close()


for view in ('top', 'wrist'):
    for fr in (39, 45):
        grab(shard / f'videos/chunk-000/observation.images.{view}/episode_{ep:06d}.mp4',
             fr, OUT / f'd2_ep{ep}_{view}_f{fr}.png')

# C3 B start frame
c3 = ROOT / 'results/experiment_2_5/G1/C3/seed_40000000'
fr_rows = list(__import__('csv').DictReader((c3 / 'frame_index.csv').open(encoding='utf-8-sig')))
b = next(x for x in fr_rows if x['skill'] == 'B')
idx = int(b['frame_index'])
print('C3 B frame', idx)
for view in ('top', 'contact'):
    grab(c3 / f'{view}.mp4', idx, OUT / f'c3_seed40000000_{view}_f{idx}.png')
