"""实验 2.1 阶段 6：B/C 原子训练状态覆盖 vs A0/A90 终态。

读取 480 集数据集的 B/C 段（observation.state），以及 40 个 A 终态快照：
  - 关节空间的分布与 PCA 投影（训练轨迹 vs A 终态 vs HOME）；
  - 每个 A 终态到 B/C 训练状态集合的最近邻距离；
  - 到 B/C episode 起始状态的最近邻距离（起始状态全是 HOME）。

输出 results/experiment_2_1/coverage/{coverage.json, pca.png}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
HOME = np.array([0.0, -0.5, 0.0, -2.1, 0.0, 1.65, 0.78])
SKILL_TASK = {
    'A0': 'pick up the orange block and place it in the target area, keeping its long side aligned with Y',
    'A90': 'pick up the orange block and place it in the target area, with its long side aligned with X',
    'B': 'push the orange block along the positive X direction to the wall',
    'C': 'pick up the orange block and place it into the tray, with its long side aligned with Y',
}


def font(size):
    for p in ('C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/simhei.ttf'):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def load_demo_states(dataset: Path, skills=('B', 'C'), stride: int = 2):
    import pyarrow.parquet as pq

    info = json.loads((dataset / 'meta/info.json').read_text(encoding='utf-8'))
    episodes = [json.loads(line) for line in
                (dataset / 'meta/episodes.jsonl').read_text(encoding='utf-8').splitlines() if line]
    tasks = {SKILL_TASK[s]: s for s in ('A0', 'A90', 'B', 'C')}
    states, starts = [], []
    per_skill = {}
    for ep in episodes:
        skill = tasks.get(ep['tasks'][0])
        if skill not in skills:
            continue
        idx = int(ep['episode_index'])
        path = dataset / info['data_path'].format(
            episode_chunk=idx // info['chunks_size'], episode_index=idx)
        table = pq.read_table(path, columns=['observation.state'])
        arr = np.asarray([row for row in table['observation.state'].to_pylist()], dtype=np.float64)
        starts.append(arr[0, :7])
        states.append(arr[::stride, :7])
        per_skill.setdefault(skill, []).append(arr[::stride, :7])
    return np.concatenate(states), np.asarray(starts), info, \
        {k: np.concatenate(v) for k, v in per_skill.items()}


def load_a_terminals(snapshots: Path):
    out = {'A0': [], 'A90': []}
    for skill in ('A0', 'A90'):
        for path in sorted(snapshots.glob(f'{skill}_*.json')):
            if 'restore' in path.name or 'collection' in path.name:
                continue
            snap = json.loads(path.read_text(encoding='utf-8'))
            out[skill].append(snap['robot']['q'])
    return {k: np.asarray(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='data/lerobot_panda_multi')
    ap.add_argument('--snapshots', default='results/experiment_2_1/snapshots')
    ap.add_argument('--out', default='results/experiment_2_1/coverage')
    ap.add_argument('--export', help='只把训练状态/起点/A 终态导出成 npz（在服务器上跑）')
    ap.add_argument('--npz', help='从导出的 npz 读取，不访问数据集')
    args = ap.parse_args()

    if args.export:
        states, starts, info, per_skill = load_demo_states(Path(args.dataset))
        terms = load_a_terminals(Path(args.snapshots))
        np.savez(args.export, states=states, starts=starts,
                 a0=terms['A0'], a90=terms['A90'],
                 b=per_skill['B'], c=per_skill['C'])
        print(f'exported {len(states)} states, {len(starts)} starts -> {args.export}')
        return

    if args.npz:
        data = np.load(args.npz)
        states, starts = data['states'], data['starts']
        terms = {'A0': data['a0'], 'A90': data['a90']}
        per_skill = {'B': data['b'], 'C': data['c']} if 'b' in data else {}
    else:
        states, starts, info, per_skill = load_demo_states(Path(args.dataset))
        terms = load_a_terminals(Path(args.snapshots))

    mean = states.mean(axis=0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    std = states.std(axis=0)
    centered = states - mean
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    pcs = vt[:2]
    proj = centered @ pcs.T
    var_ratio = (s ** 2 / (s ** 2).sum())[:2]

    def project(x):
        return (np.asarray(x, dtype=float) - mean) @ pcs.T

    def knn(points, base):
        d = np.linalg.norm(points[:, None, :] - base[None, :, :], axis=2)
        return d.min(axis=1)

    summary = {
        'demo_states': int(states.shape[0]),
        'demo_std': std.round(4).tolist(),
        'demo_mean': mean.round(4).tolist(),
        'pca_var_ratio': var_ratio.round(4).tolist(),
        'starts_unique': int(len({tuple(np.round(r, 5)) for r in starts})),
        'start_mean': starts.mean(axis=0).round(4).tolist(),
        'start_std': starts.std(axis=0).round(6).tolist(),
        'terminal_to_demo_knn_rad': {},
        'terminal_to_start_knn_rad': {},
        'terminal_to_home_rad': {},
    }
    for skill, arr in terms.items():
        summary['terminal_to_demo_knn_rad'][skill] = knn(arr, states).round(3).tolist()
        summary['terminal_to_start_knn_rad'][skill] = knn(arr, starts).round(3).tolist()
        summary['terminal_to_home_rad'][skill] = np.linalg.norm(arr - HOME, axis=1).round(3).tolist()
        summary.setdefault('terminal_mean', {})[skill] = arr.mean(axis=0).round(4).tolist()
        for other, base in per_skill.items():
            summary.setdefault(f'terminal_to_{other}_knn_rad', {})[skill] = \
                knn(arr, base).round(3).tolist()
    (out / 'coverage.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                       encoding='utf-8')

    # PCA 图
    w, h = 980, 760
    img = Image.new('RGB', (w, h), 'white')
    d = ImageDraw.Draw(img)
    d.text((16, 12), 'B/C 训练状态 vs A 终态（关节空间 PCA）', fill=(20, 20, 20), font=font(18))
    d.text((16, 38), f'训练帧 {len(states)}，方差解释 {var_ratio[0]:.0%} / {var_ratio[1]:.0%}；'
                     '高维投影不能表示完整可执行集合', fill=(90, 90, 90), font=font(12))
    left, top, right, bottom = 80, 70, w - 40, h - 70
    all_pts = np.vstack([proj,
                         project(np.asarray(terms['A0'])),
                         project(np.asarray(terms['A90'])),
                         project(HOME[None, :])])
    lo = all_pts.min(axis=0)
    hi = all_pts.max(axis=0)
    pad = (hi - lo) * 0.08
    lo -= pad
    hi += pad

    def to_px(p):
        x = left + (right - left) * (p[0] - lo[0]) / (hi[0] - lo[0])
        y = bottom - (bottom - top) * (p[1] - lo[1]) / (hi[1] - lo[1])
        return float(x), float(y)

    for p in proj[::3]:
        x, y = to_px(p)
        d.point((x, y), fill=(150, 180, 220))
    for skill, color, label in (('A0', (52, 110, 199), 'A0 终态'),
                                ('A90', (190, 70, 70), 'A90 终态')):
        for p in project(terms[skill]):
            x, y = to_px(p)
            d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=color)
    x, y = to_px(project(HOME[None, :])[0])
    d.rectangle([x - 6, y - 6, x + 6, y + 6], fill=(60, 150, 90))
    d.text((x + 10, y - 8), 'HOME（B/C 起点）', fill=(30, 110, 60), font=font(12))
    d.text((left, bottom + 12), 'PC1', fill=(80, 80, 80), font=font(12))
    d.text((left - 60, top - 4), 'PC2', fill=(80, 80, 80), font=font(12))
    d.ellipse([right - 220, top - 10, right - 208, top + 2], fill=(52, 110, 199))
    d.text((right - 200, top - 12), 'A0 终态', fill=(40, 40, 40), font=font(12))
    d.ellipse([right - 120, top - 10, right - 108, top + 2], fill=(190, 70, 70))
    d.text((right - 100, top - 12), 'A90 终态', fill=(40, 40, 40), font=font(12))
    img.save(out / 'pca.png')
    print(json.dumps({k: summary[k] for k in
                      ('demo_states', 'starts_unique', 'pca_var_ratio',
                       'terminal_to_demo_knn_rad', 'terminal_to_start_knn_rad')},
                     ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
