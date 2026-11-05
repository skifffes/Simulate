"""图像-状态同步与物体轨迹的视频级验证。

1. 同步性：对每条抽检 episode，计算相邻帧的平均像素差（top 相机）与同一帧区间的
   |Δq|（关节变化量），若图像与状态同步，两者应正相关（Spearman/Pearson）。
2. 物体轨迹（B90）：用橙色阈值在 top 相机里检测方块像素质心，跟踪其 x 方向位移，
   与严格判据记录的 advance_x_m 对齐（像素位移应显著、方向一致）。
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

SRV = Path('/scratch/lijh/exp43')
L = Path('/scratch/lijh/exp40/repo/results/experiment_4_3/data_expansion')


def orange_centroid(frame_rgb):
    r, g, b = (frame_rgb[..., 0].astype(int), frame_rgb[..., 1].astype(int),
               frame_rgb[..., 2].astype(int))
    mask = (r > 150) & (g > 50) & (g < 160) & (b < 90)
    if mask.sum() < 30:
        return None
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean()), int(mask.sum())


def sync_for(e):
    top = next((v for v in e['videos'] if 'images.top' in v), None)
    if not top:
        return None
    df = pd.read_parquet(SRV / e['parquet'])
    st = np.stack(df['observation.state'].to_numpy()).astype(float)
    dq = np.abs(np.diff(st[:, :7], axis=0)).max(axis=1)
    cap = cv2.VideoCapture(str(SRV / top))
    prev, diffs, cents = None, [], []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        c = orange_centroid(rgb)
        cents.append(c[0] if c else np.nan)
        if prev is not None:
            diffs.append(float(np.abs(rgb.astype(float) - prev.astype(float)).mean()))
        prev = rgb
    cap.release()
    n = min(len(diffs), len(dq))
    if n < 20:
        return None
    d, q = np.asarray(diffs[:n]), dq[:n]
    corr = float(np.corrcoef(d, q)[0, 1]) if d.std() > 0 and q.std() > 0 else None
    cx = np.asarray(cents[:n + 1], dtype=float)
    valid = ~np.isnan(cx)
    shift_px = float(np.nanmax(cx) - np.nanmin(cx)) if valid.sum() > 5 else None
    return {'sync_corr': corr, 'n_frames': n, 'block_px_shift': shift_px,
            'block_detected_frac': float(valid.mean())}


def main() -> int:
    man = json.loads((L / 'train_manifest_v2.json').read_text(encoding='utf-8'))
    eps = man['episodes']
    sample = eps[::max(1, len(eps) // 20)]
    rows = []
    for e in sample:
        r = sync_for(e)
        if r is None:
            continue
        r.update({'gid': e['global_episode_id'], 'skill': e['skill'],
                  'config': e['config_id'],
                  'advance_m': (e.get('metrics') or {}).get('advance_x_m')})
        rows.append(r)
    a90 = [r['sync_corr'] for r in rows if r['skill'] == 'A90' and r['sync_corr'] is not None]
    b90 = [r['sync_corr'] for r in rows if r['skill'] == 'B90' and r['sync_corr'] is not None]
    b90_shift = [r['block_px_shift'] for r in rows if r['skill'] == 'B90'
                 and r['block_px_shift'] is not None]
    out = {'n': len(rows), 'rows': rows,
           'sync_corr_A90_median': float(np.median(a90)) if a90 else None,
           'sync_corr_B90_median': float(np.median(b90)) if b90 else None,
           'B90_block_px_shift_median': float(np.median(b90_shift)) if b90_shift else None,
           'B90_block_detected_frac_median':
               float(np.median([r['block_detected_frac'] for r in rows
                                if r['skill'] == 'B90']))}
    (L / 'image_sync_check.json').write_text(json.dumps(out, ensure_ascii=False, indent=2,
                                                        default=float), encoding='utf-8')
    print(json.dumps({k: v for k, v in out.items() if k != 'rows'}, ensure_ascii=False, indent=2))
    print('per-episode (skill, sync_corr, px_shift, detected_frac):')
    for r in rows:
        print(f"  {r['skill']} gid{r['gid']:3d} corr={None if r['sync_corr'] is None else round(r['sync_corr'], 2)} "
              f"px={None if r['block_px_shift'] is None else round(r['block_px_shift'], 1)} "
              f"det={round(r['block_detected_frac'], 2)}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
