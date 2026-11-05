"""补齐视频抽查缺的那个训练条件：找出未覆盖的 config，取一条 episode 生成 montage 并统计。"""
import json
from pathlib import Path

import cv2
import numpy as np

SRV = Path('/scratch/lijh/exp43')
L = Path('/scratch/lijh/exp40/repo/results/experiment_4_3/data_expansion')
OUT = SRV / 'video_check'

man = json.loads((L / 'train_manifest_v2.json').read_text(encoding='utf-8'))
check = json.loads((OUT / 'video_spotcheck.json').read_text(encoding='utf-8'))
covered = {r['config'] for r in check['rows']}
all_cfgs = {e['config_id'] for e in man['episodes']}
missing = sorted(all_cfgs - covered)
print('missing configs:', missing)
rows = []
for cfg in missing:
    e = next(x for x in man['episodes'] if x['config_id'] == cfg)
    vids = e['videos']
    top = next(v for v in vids if 'images.top' in v)
    wrist = next((v for v in vids if 'wrist' in v), None)
    for label, v in (('top', top), ('wrist', wrist)):
        if not v:
            continue
        cap = cv2.VideoCapture(str(SRV / v))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        for frac in (0.25, 0.5, 0.75):
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(n - 1, int(n * frac))))
            ok, f = cap.read()
            if ok:
                frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cap.release()
        if frames:
            cv2.imwrite(str(OUT / f"gid{e['global_episode_id']:03d}_{e['skill']}_{label}.png"),
                        cv2.cvtColor(np.concatenate(frames, axis=1), cv2.COLOR_RGB2BGR))
            rows.append({'config': cfg, 'gid': e['global_episode_id'], 'camera': label,
                         'n_frames': n,
                         'frame_means': [round(float(f.mean()), 1) for f in frames],
                         'frame_stds': [round(float(f.std()), 1) for f in frames]})
        print('wrote', label, e['skill'], cfg)
(L / 'video_spotcheck_extra.json').write_text(
    json.dumps({'missing_configs': missing, 'rows': rows}, ensure_ascii=False, indent=2),
    encoding='utf-8')
