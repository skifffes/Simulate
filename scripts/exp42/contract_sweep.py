"""Gate 4 任务 3（训练前）：两种表示的"归一化→反归一化→解码→物理目标"闭环契约测试。

不训练、不用 GPU。验证四件事：
  1. quantile 归一化/反归一化往返误差（两种表示的目标分布各自）；
  2. 模型噪声在**物理目标**上的传播：anchor（每行独立）vs target_increment（chunk 内累积）；
  3. 安全裁剪在两种表示下的触发率（fixed: 0.25*(k+1)；recursive: 0.25/步）；
  4. 正常（无噪声）时两种解码都精确重建专家命令（回归）。

用法（服务器 simulate venv）：
  python scripts/exp42/contract_sweep.py \
    --stats /scratch/lijh/.cache/openpi/openpi-assets/checkpoints/pi05_droid/assets/droid/norm_stats.json \
    --episode B90:0 --out results/experiment_4_2/contract_replay_p0c/contract_sweep.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / 'src', ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

DATA = Path('/scratch/lijh/exp42/data')
FINGER_MAX_M = 0.04
HORIZON = 16
MAX_DELTA = 0.25


def load_episode(skill: str, ep: int):
    meta = [json.loads(l) for l in (DATA / 'meta.jsonl').read_text(encoding='utf-8').splitlines()
            if l.strip()]
    row = next(m for m in meta if m['skill'] == skill and m['episode_index'] == ep)
    z = np.load(DATA / row['state_path'])
    return row, z['state'].astype(np.float64), z['action'].astype(np.float64)


def action_stats(path: Path):
    """递归找 ``actions`` 的 q01/q99（DROID 资产结构：norm_stats/{actions,state}/{q01,q99}）。"""
    d = json.loads(path.read_text(encoding='utf-8'))

    def find(o):
        if isinstance(o, dict):
            for key in ('actions', 'action'):
                v = o.get(key)
                if isinstance(v, dict) and 'q01' in v and 'q99' in v:
                    return np.asarray(v['q01'], float), np.asarray(v['q99'], float)
            for v in o.values():
                got = find(v)
                if got is not None:
                    return got
        return None

    got = find(d)
    if got is None:
        raise ValueError('no actions q01/q99 in stats')
    return got


def to_norm(x, q01, q99):
    return 2.0 * (x - q01) / np.maximum(q99 - q01, 1e-9) - 1.0


def from_norm(u, q01, q99):
    return (u + 1.0) * np.maximum(q99 - q01, 1e-9) / 2.0 + q01


def clip_rows(delta, mode, k_index):
    """fixed: 第 k 行上限 0.25*(k+1)；recursive: 每步 0.25。"""
    out = np.clip(delta, -MAX_DELTA, MAX_DELTA) if mode == 'recursive' else \
        np.clip(delta, -MAX_DELTA * (k_index + 1), MAX_DELTA * (k_index + 1))
    return out


def decode(labels, q0, mode):
    """把 16×7 的（物理）标签解码成物理关节目标。"""
    if mode == 'fixed':
        return q0[None, :] + labels
    return q0[None, :] + np.cumsum(labels, axis=0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--stats', required=True)
    ap.add_argument('--episode', default='B90:0')
    ap.add_argument('--start', type=int, default=20)
    ap.add_argument('--frames', type=int, default=4,
                    help='用连续若干帧的窗口做噪声扫描（每个窗口一次）')
    ap.add_argument('--noise', default='0,0.01,0.05,0.1,0.2')
    ap.add_argument('--repeats', type=int, default=32)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)

    skill, ep = args.episode.split(':')
    _, S, A = load_episode(skill, int(ep))
    q01, q99 = action_stats(Path(args.stats))
    rng = np.random.default_rng(args.seed)
    sigmas = [float(x) for x in args.noise.split(',')]

    res = {'episode': args.episode, 'start': args.start, 'horizon': HORIZON,
           'sigmas': sigmas, 'repeats': args.repeats,
           'stats': str(args.stats), 'rows': []}

    for offset in range(args.frames):
        t = args.start + offset
        q0 = S[t, :7].copy()
        tgt = A[t:t + HORIZON, :7]
        expert_anchor = tgt - q0[None, :]
        expert_inc = np.empty_like(expert_anchor)
        expert_inc[0] = tgt[0] - q0
        expert_inc[1:] = tgt[1:] - tgt[:-1]
        expert_targets = tgt

        entry = {'t': t, 'noise': {}}
        for name, labels in (('anchor', expert_anchor), ('target_increment', expert_inc)):
            u = to_norm(labels, q01[:7], q99[:7])
            back = from_norm(u, q01[:7], q99[:7])
            entry.setdefault('round_trip', {})[name] = float(np.abs(back - labels).max())
            entry.setdefault('normalized_span', {})[name] = {
                'min': float(u.min()), 'max': float(u.max()),
                'frac_outside_pm1': float((np.abs(u) > 1).mean())}

        for sigma in sigmas:
            err_fixed, err_rec, clip_fixed, clip_rec = [], [], [], []
            for _ in range(args.repeats if sigma > 0 else 1):
                for name, labels, mode in (
                        ('anchor', expert_anchor, 'fixed'),
                        ('increment', expert_inc, 'recursive')):
                    u = to_norm(labels, q01[:7], q99[:7])
                    if sigma > 0:
                        u = u + rng.normal(0.0, sigma, size=u.shape)
                    phys = from_norm(u, q01[:7], q99[:7])
                    if mode == 'fixed':
                        clipped = np.stack([clip_rows(phys[k], 'fixed', k)
                                            for k in range(HORIZON)]) if False else \
                            np.clip(phys, -MAX_DELTA * (np.arange(HORIZON)[:, None] + 1),
                                    MAX_DELTA * (np.arange(HORIZON)[:, None] + 1))
                    else:
                        clipped = np.clip(phys, -MAX_DELTA, MAX_DELTA)
                    target = decode(clipped, q0, 'fixed' if mode == 'fixed' else 'recursive')
                    err = float(np.abs(target - expert_targets).max())
                    n_clip = int((np.abs(phys) > np.abs(clipped) + 1e-12).sum())
                    if mode == 'fixed':
                        err_fixed.append(err)
                        clip_fixed.append(n_clip)
                    else:
                        err_rec.append(err)
                        clip_rec.append(n_clip)
            entry['noise'][str(sigma)] = {
                'anchor_fixed_max_err_rad_mean': float(np.mean(err_fixed)),
                'anchor_fixed_max_err_rad_p95': float(np.percentile(err_fixed, 95)),
                'increment_recursive_max_err_rad_mean': float(np.mean(err_rec)),
                'increment_recursive_max_err_rad_p95': float(np.percentile(err_rec, 95)),
                'anchor_clipped_values_mean': float(np.mean(clip_fixed)),
                'increment_clipped_values_mean': float(np.mean(clip_rec)),
            }
        res['rows'].append(entry)
        print(f"t={t}: round_trip anchor={entry['round_trip']['anchor']:.2e} "
              f"inc={entry['round_trip']['target_increment']:.2e} | "
              f"err@sigma0.1 anchor={entry['noise']['0.1']['anchor_fixed_max_err_rad_mean']:.3f} "
              f"inc={entry['noise']['0.1']['increment_recursive_max_err_rad_mean']:.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print('wrote', args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
