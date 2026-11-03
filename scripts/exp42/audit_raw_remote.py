"""Read-only raw/cache provenance, startup, and alignment audit; no model loading."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    meta = [json.loads(x) for x in (args.data / 'meta.jsonl').read_text().splitlines() if x.strip()]
    summary = {'python': sys.executable, 'episodes': len(meta), 'files': {}, 'startup': []}
    for rel in ['scripts/exp42/train_lora.py', 'scripts/exp42/eval_rollouts.py',
                'src/exp40/action_adapter.py', 'src/vla/atomic_tasks/recording.py',
                'src/exp41/experts41.py']:
        path = args.root / rel
        summary['files'][rel] = digest(path) if path.exists() else 'MISSING'
    summary['openpi_head'] = subprocess.run(
        ['git', '-C', '/scratch/lijh/exp40/openpi', 'rev-parse', 'HEAD'],
        capture_output=True, text=True).stdout.strip()
    first_rows, sync_rows = [], []
    import pandas as pd
    import imageio.v3 as iio
    for row in meta:
        skill, ep = row['skill'], row['episode_index']
        with np.load(args.data / row['state_path'], allow_pickle=False) as z:
            s, a = z['state'], z['action']
        dq = a[:, :7] - s[:, :7]
        actual = np.diff(s[:, :7], axis=0)
        moving = np.flatnonzero(np.max(np.abs(dq), axis=1) > 1e-4)
        entry = {'skill': skill, 'episode': ep, 'split': row['split'], 'n_frames': len(s),
                 'first_command_error_gt_1e4': int(moving[0]) if len(moving) else None,
                 'first_actual_motion_gt_1e4': next((int(t) for t in range(len(actual))
                     if np.max(np.abs(actual[t])) > 1e-4), None),
                 'first_q': s[0, :7].tolist(), 'first_target': a[0, :7].tolist(),
                 'first_gripper_target_m': float(a[0, 7]),
                 'state_sha256': digest(args.data / row['state_path'])}
        summary['startup'].append(entry)
        if ep >= 3:
            continue
        raw = args.root / 'data/exp41' / skill
        parq = raw / 'data' / f'chunk-{ep // 1000:03d}' / f'episode_{ep:06d}.parquet'
        if not parq.exists():
            sync_rows.append({'skill': skill, 'episode': ep, 'status': 'RAW_PARQUET_MISSING', 'path': str(parq)})
            continue
        df = pd.read_parquet(parq)
        rs = np.stack(df['observation.state'])
        ra = np.stack(df['action'])
        sync_rows.append({'skill': skill, 'episode': ep, 'status': 'RAW_CACHE_COMPARED',
                          'state_equal': bool(np.array_equal(rs, s)),
                          'action_equal': bool(np.array_equal(ra, a)),
                          'raw_rows': len(df), 'cache_rows': len(s),
                          'timestamp_first40': df['timestamp'].iloc[:40].tolist() if 'timestamp' in df else None,
                          'parquet_sha256': digest(parq)})
        for t in range(min(40, len(s) - 1)):
            first_rows.append([skill, ep, t, t / 20, *[json.dumps(x.tolist()) for x in
                (s[t], a[t], s[t+1], dq[t], actual[t], a[t, :7]-s[t+1, :7])]])
        for camera, key in [('top', 'observation.images.top'), ('wrist', 'observation.images.wrist')]:
            video = raw / 'videos' / f'chunk-{ep // 1000:03d}' / key / f'episode_{ep:06d}.mp4'
            cached = np.load(args.data / row[f'{camera}_path'], mmap_mode='r')
            decoded = []
            for t, image in enumerate(iio.imiter(video)):
                decoded.append(image)
                if t >= 39:
                    break
            frames = np.stack(decoded)
            sync_rows.append({'skill': skill, 'episode': ep, 'camera': camera,
                              'status': 'DECODED_FIRST40_COMPARED', 'cache_shape': list(cached.shape),
                              'decoded_shape': list(frames.shape),
                              'same_index_pixel_equal': bool(np.array_equal(frames, cached[:len(frames)])),
                              'video_sha256': digest(video),
                              'first_frame_sha256': hashlib.sha256(frames[0].tobytes()).hexdigest(),
                              'first40_mean_abs_adjacent_pixel_difference': float(np.abs(np.diff(frames.astype(np.float32), axis=0)).mean())})
    with (args.out / 'raw_first40.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['skill', 'episode', 'frame', 'time_s', 'state', 'target', 'next_state',
                    'target_minus_q', 'actual_delta', 'tracking_residual'])
        w.writerows(first_rows)
    summary['synchronization'] = sync_rows
    summary['limits'] = ['Array/video equality establishes exporter index preservation, not independent proof of capture timestamps.',
                         'No policy inference, training or physics evaluation performed.']
    (args.out / 'raw_audit.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({'episodes': len(meta), 'first40_rows': len(first_rows), 'sync_checks': len(sync_rows), 'out': str(args.out)}))


if __name__ == '__main__':
    main()
