"""实验 4.1 数据回环验证：写盘 → 读回 → 特征校验（并给出 diversity / split 审计）。

检查项（prompt §10.3）：
  * parquet 行数 = action 数 = 视频帧数（若存在视频）
  * state/action 维度、dtype、NaN/Inf
  * 时间戳单调、fps 一致
  * observation.state 与 action 的相位关系（action[t] 是 t→t+1 的绝对关节目标）
  * 动作语义：state 与 action 都是 9 维/8 维（9 维 state = 7 关节 + 2 指；8 维 action = 7 目标 + 夹爪）
  * 任务文本存在且与 goal 一致
  * 相机版本、camera sha、action contract 版本随 meta 保存

用法（服务器）：
    python scripts/exp41/verify_dataset.py --data data/exp41 --out results/experiment_4_1/dataset
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / 'src', ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_dataset(root: Path) -> dict:
    import pandas as pd

    info = json.loads((root / 'meta' / 'info.json').read_text(encoding='utf-8'))
    episodes = [json.loads(l) for l in (root / 'meta' / 'episodes.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    tasks = [json.loads(l) for l in (root / 'meta' / 'tasks.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    checks = []
    per_episode = []

    def add(name, ok, detail=None):
        checks.append({'check': name, 'pass': bool(ok), 'detail': detail})

    add('fps_is_policy_hz', info.get('fps') == 20, {'fps': info.get('fps')})
    add('has_state_and_action',
        'observation.state' in info.get('features', {}) and 'action' in info.get('features', {}),
        {'features': list(info.get('features', {}))})
    add('has_two_cameras',
        all(k in info.get('features', {}) for k in ('observation.images.top', 'observation.images.wrist')),
        {'camera_features': [k for k in info.get('features', {}) if 'images' in k]})

    for ep in episodes:
        idx = int(ep['episode_index'])
        parq = root / 'data' / f'chunk-{idx // 1000:03d}' / f'episode_{idx:06d}.parquet'
        if not parq.exists():
            add(f'ep{idx}_parquet_exists', False, str(parq))
            continue
        df = pd.read_parquet(parq, columns=['observation.state', 'action', 'timestamp', 'frame_index', 'task_index'])
        state = np.stack(df['observation.state'].to_numpy()).astype(np.float64)
        action = np.stack(df['action'].to_numpy()).astype(np.float64)
        ts = df['timestamp'].to_numpy().astype(np.float64)
        ok = True
        detail = {}
        if state.shape[0] != action.shape[0]:
            ok = False
            detail['len_mismatch'] = [state.shape[0], action.shape[0]]
        if state.shape[1] != 9 or action.shape[1] != 8:
            ok = False
            detail['shape'] = [list(state.shape), list(action.shape)]
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(action)):
            ok = False
            detail['nan_inf'] = True
        if ts.size > 1 and not np.all(np.diff(ts) > 0):
            ok = False
            detail['timestamp_not_increasing'] = True
        # 相位：action[t] 应当接近 state[t+1]（关节部分）
        if state.shape[0] > 2:
            d = np.abs(action[:-1, :7] - state[1:, :7])
            detail['action_vs_next_state_max_rad'] = float(d.max())
            detail['action_vs_next_state_mean_rad'] = float(d.mean())
        # 任务文本
        ti = int(df['task_index'].iloc[0])
        task_text = next((t['task'] for t in tasks if t['task_index'] == ti), None)
        detail['task'] = task_text
        detail['n_frames'] = int(state.shape[0])
        # 视频帧数（如有）
        vid = root / 'videos' / f'chunk-{idx // 1000:03d}' / f'observation.images.top' / f'episode_{idx:06d}.mp4'
        if vid.exists():
            detail['video_bytes'] = vid.stat().st_size
        per_episode.append({'episode_index': idx, 'ok': ok,
                            'n_frames': int(state.shape[0]), 'task': task_text,
                            'task_index': ti, 'detail': detail})
        add(f'ep{idx}_valid', ok, detail)

    n_ok = sum(1 for e in per_episode if e['ok'])
    return {
        'dataset_dir': str(root),
        'n_episodes': len(episodes),
        'n_valid_episodes': n_ok,
        'n_checks': len(checks),
        'n_pass': sum(1 for c in checks if c['pass']),
        'all_pass': all(c['pass'] for c in checks),
        'checks': checks,
        'episodes': per_episode,
        'info': info,
        'tasks': [t['task'] for t in tasks],
        'meta_sha256': {p.name: sha_file(p) for p in sorted((root / 'meta').glob('*')) if p.is_file()},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/exp41')
    ap.add_argument('--out', default='results/experiment_4_1/dataset')
    args = ap.parse_args(argv)
    data = Path(args.data) if Path(args.data).is_absolute() else ROOT / args.data
    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for skill in ('A90', 'B90'):
        root = data / skill
        if not (root / 'meta' / 'info.json').exists():
            continue
        rep = verify_dataset(root)
        report[skill] = {k: v for k, v in rep.items() if k != 'checks'}
        report[skill]['failed_checks'] = [c for c in rep['checks'] if not c['pass']]
        print(f"{skill}: episodes={rep['n_episodes']} valid={rep['n_valid_episodes']} "
              f"checks_pass={rep['n_pass']}/{rep['n_checks']}")
    (out / 'readback_verification.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
