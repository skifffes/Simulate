"""把多组 eval_vla_policy.py 的 eval.json 汇总成 markdown 表格。

用法（可在服务器或本地运行）：
    python server/collect_eval_table.py name=path/eval.json [name=path/eval.json ...]

每组输出一行：episodes / success / success_rate / 失败分布 / 失败阶段 /
平均抬升高度 / 平均放置误差 / 平均 tick / 平均跟踪误差 / 推理延迟。
同时打印逐 episode 失败 seed 清单，便于在报告里如实披露。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


COLS = [
    ('episodes', 'episodes'),
    ('success', 'success'),
    ('success_rate', '成功率'),
    ('failure_counts', '失败类型'),
    ('failure_stage_counts', '失败阶段'),
    ('mean_lift_height_m', '平均抬升/m'),
    ('mean_place_xy_error_m', '放置误差/m'),
    ('mean_ticks', '平均tick'),
    ('mean_tracking_error_rad', '跟踪误差/rad'),
    ('mean_inference_latency_ms', '推理延迟/ms'),
]


def _fmt(key: str, value) -> str:
    if value is None:
        return '-'
    if key in ('success_rate',):
        return f'{float(value) * 100:.1f}%'
    if key == 'failure_counts':
        return ', '.join(f'{k}:{v}' for k, v in value.items()) or '-'
    if key == 'failure_stage_counts':
        return ', '.join(f'{k}:{v}' for k, v in value.items()) or '-'
    if key.startswith('mean_') and isinstance(value, (int, float)):
        return f'{float(value):.4g}'
    return str(value)


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    rows = []
    details = []
    for arg in argv:
        name, _, path = arg.partition('=')
        if not path:
            print(f'参数需要 name=path 形式: {arg}')
            return 2
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        summary = data.get('summary', data)
        episodes = data.get('episodes', [])
        failed = [ep.get('seed') for ep in episodes if not ep.get('success')]
        rows.append((name, summary))
        details.append((name, failed, episodes))

    header = ['组别'] + [label for _, label in COLS]
    print('| ' + ' | '.join(header) + ' |')
    print('|' + '|'.join(['---'] * len(header)) + '|')
    for name, summary in rows:
        cells = [name] + [_fmt(key, summary.get(key)) for key, _ in COLS]
        print('| ' + ' | '.join(cells) + ' |')

    print()
    for name, failed, episodes in details:
        walls = [ep.get('wall_seconds') for ep in episodes if ep.get('wall_seconds')]
        sims = [ep.get('sim_seconds') for ep in episodes if ep.get('sim_seconds')]
        lat = [ep.get('inference_latency_ms_mean') for ep in episodes
               if ep.get('inference_latency_ms_mean')]
        extra = ''
        if walls:
            extra += f' 平均墙钟 {sum(walls) / len(walls):.1f}s'
        if sims:
            extra += f' 平均仿真 {sum(sims) / len(sims):.1f}s'
        if lat:
            extra += f' 推理 {sum(lat) / len(lat):.1f}ms'
        print(f'- {name}: 失败 seed {failed or "无"}{extra}')
        stages = [ep.get('failure_stage') for ep in episodes if not ep.get('success')]
        if stages:
            print(f'  失败阶段明细: {stages}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
