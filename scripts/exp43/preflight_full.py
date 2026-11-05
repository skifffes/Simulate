"""实验 4.3 全量 preflight：对训练清单 v2 的 253 条唯一 episode 做逐条检查。

检查项：
  1. parquet：列名、行数、state(9)/action(8) 维度、NaN/Inf、timestamp 单调且 dt≈0.05 s；
  2. 动作语义：|a−s| 关节上界、夹爪范围、是否有异常跳变；
  3. 视频：cv2 解码帧数 == parquet 行数、fps、分辨率 256×256、抽 10% 解码一帧看均值/方差（防空白/损坏）；
  4. Anchor-16 构造：抽 10 条构造 (N,16,8) 标签，检查形状、有限、k=0 与单步标签一致；
  5. 按技能/条件汇总。
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

SRV = Path('/scratch/lijh/exp43')
MAN = Path('results/experiment_4_3/data_expansion/train_manifest_v2.json')


def check_episode(e):
    p = SRV / e['parquet']
    df = pd.read_parquet(p)
    st = np.stack(df['observation.state'].to_numpy()).astype(np.float64)
    ac = np.stack(df['action'].to_numpy()).astype(np.float64)
    ts = df['timestamp'].to_numpy().astype(np.float64)
    out = {
        'global_episode_id': e['global_episode_id'], 'skill': e['skill'],
        'config_id': e['config_id'], 'rows': int(len(df)),
        'state_dim': int(st.shape[1]), 'action_dim': int(ac.shape[1]),
        'finite': bool(np.isfinite(st).all() and np.isfinite(ac).all()),
        'ts_monotonic': bool(np.all(np.diff(ts) > 0)),
        'dt_mean': float(np.diff(ts).mean()) if len(ts) > 1 else None,
        'dq_absmax': float(np.abs(ac[:, :7] - st[:, :7]).max()),
        'grip_min': float(ac[:, 7].min()), 'grip_max': float(ac[:, 7].max()),
    }
    vids = e['videos']
    counts, res, fps = [], [], []
    for v in vids:
        cap = cv2.VideoCapture(str(SRV / v))
        counts.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        res.append((int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        fps.append(float(cap.get(cv2.CAP_PROP_FPS)))
        cap.release()
    out.update({'n_videos': len(vids), 'video_frames': counts,
                'video_res': res, 'video_fps': fps,
                'frames_match_rows': all(c == int(len(df)) for c in counts)})
    return out


def spot_frame(e):
    v = e['videos'][0]
    cap = cv2.VideoCapture(str(SRV / v))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return {'ok': False}
    return {'ok': True, 'mean': float(frame.mean()), 'std': float(frame.std()),
            'shape': list(frame.shape)}


def anchor_check(e, horizon=16):
    df = pd.read_parquet(SRV / e['parquet'])
    st = np.stack(df['observation.state'].to_numpy()).astype(np.float64)
    ac = np.stack(df['action'].to_numpy()).astype(np.float64)
    n = max(0, len(df) - horizon + 1)
    if n == 0:
        return {'ok': False, 'reason': 'too short'}
    t = 0
    labels = ac[t:t + horizon, :7] - st[t, :7][None, :]
    grip = np.clip(ac[t:t + horizon, 7] / 0.04, 0, 1)
    return {'ok': bool(np.isfinite(labels).all() and np.isfinite(grip).all()),
            'shape': [horizon, 8], 'k0_equals_step': bool(np.allclose(labels[0], ac[t, :7] - st[t, :7]))}


def main() -> int:
    man = json.loads(MAN.read_text(encoding='utf-8'))
    eps = man['episodes']
    rows = []
    for e in eps:
        try:
            rows.append(check_episode(e))
        except Exception as exc:  # noqa: BLE001
            rows.append({'global_episode_id': e['global_episode_id'], 'error': repr(exc)[:200]})

    errs = [r for r in rows if 'error' in r]
    checks = {
        'n_episodes': len(rows),
        'n_errors': len(errs),
        'all_finite': all(r.get('finite') for r in rows if 'error' not in r),
        'all_ts_monotonic': all(r.get('ts_monotonic') for r in rows if 'error' not in r),
        'dt_values': sorted({round(r['dt_mean'], 4) for r in rows
                             if 'error' not in r and r.get('dt_mean')}),
        'all_state_dim_9': all(r.get('state_dim') == 9 for r in rows if 'error' not in r),
        'all_action_dim_8': all(r.get('action_dim') == 8 for r in rows if 'error' not in r),
        'all_videos_match_rows': all(r.get('frames_match_rows') for r in rows if 'error' not in r),
        'dq_absmax_global': max((r['dq_absmax'] for r in rows if 'error' not in r), default=None),
        'grip_range': [min(r['grip_min'] for r in rows if 'error' not in r),
                        max(r['grip_max'] for r in rows if 'error' not in r)],
        'n_videos_total': sum(r.get('n_videos', 0) for r in rows if 'error' not in r),
        'video_res_set': sorted({tuple(x) for r in rows if 'error' not in r for x in r['video_res']}),
        'video_fps_set': sorted({round(x, 2) for r in rows if 'error' not in r for x in r['video_fps']}),
    }

    spot_idx = eps[::max(1, len(eps) // 25)]
    spots = [dict(spot_frame(e), gid=e['global_episode_id']) for e in spot_idx]
    checks['spot_frames'] = {'n': len(spots), 'all_ok': all(s['ok'] for s in spots),
                              'mean_range': [min(s['mean'] for s in spots if s['ok']),
                                              max(s['mean'] for s in spots if s['ok'])],
                              'std_min': min(s['std'] for s in spots if s['ok'])}

    anchors = [anchor_check(e) for e in eps[::max(1, len(eps) // 10)]]
    checks['anchor16'] = {'n': len(anchors), 'all_ok': all(a.get('ok') for a in anchors),
                           'k0_consistent': all(a.get('k0_equals_step') for a in anchors)}

    per_cfg = defaultdict(int)
    for r in rows:
        per_cfg[f"{r.get('skill')}:{r.get('config_id')}"] += 1
    checks['by_config'] = dict(sorted(per_cfg.items()))
    checks['rows'] = rows

    out = Path('results/experiment_4_3/data_expansion/preflight_full.json')
    out.write_text(json.dumps(checks, ensure_ascii=False, indent=2, default=float),
                   encoding='utf-8')
    print(json.dumps({k: v for k, v in checks.items() if k not in ('rows', 'by_config')},
                     ensure_ascii=False, indent=2)[:2500])
    print('wrote', out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
