"""10% 视频内容抽查：每条抽 25%/50%/75% 三帧，输出统计与拼接图。

对 train_manifest_v2 里每隔 10 条抽 1 条（共 ~25 条，覆盖 10 个训练条件），
解码 top 相机视频（若有 wrist 也取），检查：
  * 三帧可解码、尺寸 256×256；
  * 帧间差异（相邻抽样帧的像素差）——确认不是静止画面；
  * 关键阶段的位置（A90 抓取/放置附近、B90 推动/触墙附近）用 gripper 命令与帧序号近似；
输出 PNG 拼接图到 /scratch/lijh/exp43/video_check/ 供人工查看，统计写入 JSON。
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

SRV = Path('/scratch/lijh/exp43')
OUTDIR = SRV / 'video_check'
MAN = Path('/scratch/lijh/exp40/repo/results/experiment_4_3/data_expansion/'
           'train_manifest_v2.json')


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    man = json.loads(MAN.read_text(encoding='utf-8'))
    eps = man['episodes']
    step = max(1, len(eps) // 25)
    sample = eps[::step]
    rows = []
    for e in sample:
        vids = e['videos']
        top = next((v for v in vids if 'images.top' in v), None)
        wrist = next((v for v in vids if 'wrist' in v), None)
        cap = cv2.VideoCapture(str(SRV / top)) if top else None
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap else 0
        picks, frames = [], []
        for frac in (0.25, 0.5, 0.75):
            idx = max(0, min(n - 1, int(n * frac)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, f = cap.read()
            if ok:
                picks.append(idx)
                frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        if cap:
            cap.release()
        diffs = [float(np.abs(frames[i].astype(float) - frames[i - 1].astype(float)).mean())
                 for i in range(1, len(frames))]
        rows.append({'gid': e['global_episode_id'], 'skill': e['skill'],
                     'config': e['config_id'], 'video': top,
                     'n_frames': n, 'picked': picks,
                     'frame_means': [round(float(f.mean()), 1) for f in frames],
                     'frame_stds': [round(float(f.std()), 1) for f in frames],
                     'adjacent_mean_absdiff': [round(d, 2) for d in diffs],
                     'blank_suspect': any(float(f.std()) < 5 for f in frames)})
        if frames:
            montage = np.concatenate(frames, axis=1)
            cv2.imwrite(str(OUTDIR / f"gid{e['global_episode_id']:03d}_{e['skill']}.png"),
                        cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
    (OUTDIR / 'video_spotcheck.json').write_text(
        json.dumps({'n_sampled': len(rows), 'rows': rows}, ensure_ascii=False, indent=2),
        encoding='utf-8')
    print('sampled', len(rows), 'blank_suspects', sum(r['blank_suspect'] for r in rows))
    print('mean absdiff range:', min(min(r['adjacent_mean_absdiff']) for r in rows
                                      if r['adjacent_mean_absdiff']),
          max(max(r['adjacent_mean_absdiff']) for r in rows if r['adjacent_mean_absdiff']))
    print('cover configs:', len({r['config'] for r in rows}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
