"""汇总实验 1 的评测结果，生成 results/experiment_1/summary.md 与 summary.json。

读取 results/experiment_1/eval/<组别>/eval.json，输出：
  * 各组成功率与失败类型总览
  * 同一种子下的成功/失败交叉表（chunk50 vs n10）
  * 逐 episode 明细
  * 推理延迟
用法：
    python server/make_experiment_1_summary.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / 'results' / 'experiment_1'
EVAL_DIR = RESULT_DIR / 'eval'

GROUPS = [
    ('vla_id_chunk', 'VLA 分布内 chunk50'),
    ('vla_id_n10', 'VLA 分布内 n10'),
    ('vla_ood_chunk', 'VLA 留出 chunk50'),
    ('vla_ood_n10', 'VLA 留出 n10'),
    ('expert_ood', '专家 留出'),
]


def load(name: str) -> dict:
    path = EVAL_DIR / name / 'eval.json'
    return json.loads(path.read_text(encoding='utf-8'))


def main() -> int:
    data = {name: load(name) for name, _ in GROUPS}

    lines = ['# 实验 1 评测汇总', '']
    lines.append('判据：工件抬升 ≥ 3 cm、放到目标 2 cm 内、无滑落（≤1.5 cm）、'
                 '静置时关节速度收敛且已松爪。每个 episode 最长 260 个控制周期（13 s 仿真）。')
    lines.append('')

    lines.append('## 1. 各组结果')
    lines.append('')
    lines.append('| 组别 | 集数 | 成功 | 成功率 | 失败类型 | 平均抬升/m | 平均放置误差/m | 平均 tick |')
    lines.append('|---|---|---|---|---|---|---|---|')
    for name, label in GROUPS:
        s = data[name]['summary']
        fails = ', '.join(f'{k}×{v}' for k, v in s.get('failure_counts', {}).items()) or '-'
        lines.append(
            f"| {label} | {s['episodes']} | {s['success']} | {s['success_rate'] * 100:.0f}% | "
            f"{fails} | {s['mean_lift_height_m']:.3f} | {s['mean_place_xy_error_m']:.3f} | "
            f"{s['mean_ticks']:.0f} |"
        )
    lines.append('')

    lines.append('## 2. 同种子交叉表（留出集，20 个种子）')
    lines.append('')
    chunk = {e['seed']: e['success'] for e in data['vla_ood_chunk']['episodes']}
    n10 = {e['seed']: e['success'] for e in data['vla_ood_n10']['episodes']}
    both_ok = sum(1 for s in chunk if chunk[s] and n10.get(s))
    chunk_only = sum(1 for s in chunk if chunk[s] and not n10.get(s))
    n10_only = sum(1 for s in chunk if not chunk[s] and n10.get(s))
    both_fail = sum(1 for s in chunk if not chunk[s] and not n10.get(s))
    lines.append('| chunk50 \\ n10 | n10 成功 | n10 失败 |')
    lines.append('|---|---|---|')
    lines.append(f'| chunk50 成功 | {both_ok} | {chunk_only} |')
    lines.append(f'| chunk50 失败 | {n10_only} | {both_fail} |')
    lines.append('')

    lines.append('## 3. 留出集 n10 逐 episode 明细')
    lines.append('')
    lines.append('| seed | 结果 | 失败类型 | 失败阶段 | 抬升/m | 放置误差/m | tick | 墙钟/s |')
    lines.append('|---|---|---|---|---|---|---|---|')
    for e in data['vla_ood_n10']['episodes']:
        lines.append(
            f"| {e['seed']} | {'成功' if e['success'] else '失败'} | {e.get('failure') or '-'} | "
            f"{e.get('failure_stage') or '-'} | {e['lift_height_m']:.3f} | "
            f"{e['place_xy_error_m']:.3f} | {e['ticks']} | {e['wall_seconds']:.1f} |"
        )
    lines.append('')

    latency_path = RESULT_DIR / 'latency.json'
    if latency_path.exists():
        lat = json.loads(latency_path.read_text(encoding='utf-8'))
        lines.append('## 4. 推理延迟（A40，batch=1，n_action_steps=%d）' % lat['n_action_steps'])
        lines.append('')
        lines.append('| 指标 | 数值 |')
        lines.append('|---|---|')
        lines.append(f"| 新 chunk 前向 mean | {lat['inference_ms_mean']:.1f} ms |")
        lines.append(f"| 新 chunk 前向 p50 | {lat['inference_ms_p50']:.1f} ms |")
        lines.append(f"| 新 chunk 前向 p95 | {lat['inference_ms_p95']:.1f} ms |")
        lines.append(f"| chunk 内取动作 mean | {lat['cached_action_ms_mean']:.2f} ms |")
        lines.append('')
        lines.append(
            f"按 n10 摊到每个控制周期，平均 ≈ {lat['inference_ms_mean'] / 10 + lat['cached_action_ms_mean']:.1f} ms，"
            '低于 20 Hz 的 50 ms 预算；单次前向 288 ms 无法直接 20 Hz 同步执行。'
        )
        lines.append('')

    videos = sorted((RESULT_DIR / 'videos').glob('*.mp4')) if (RESULT_DIR / 'videos').exists() else []
    if videos:
        lines.append('## 5. 视频证据（results/experiment_1/videos）')
        lines.append('')
        for v in videos:
            lines.append(f'- `{v.name}`（{v.stat().st_size / 1024:.0f} KB）')
        lines.append('')

    out_md = RESULT_DIR / 'summary.md'
    out_md.write_text('\n'.join(lines), encoding='utf-8')

    summary = {
        'groups': {
            name: {
                'label': label,
                'summary': data[name]['summary'],
                'failed_seeds': [e['seed'] for e in data[name]['episodes'] if not e['success']],
            }
            for name, label in GROUPS
        },
        'ood_cross_tab': {
            'both_ok': both_ok,
            'chunk_only_ok': chunk_only,
            'n10_only_ok': n10_only,
            'both_fail': both_fail,
        },
    }
    out_json = RESULT_DIR / 'summary.json'
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(out_md.read_text(encoding='utf-8'))
    print(f'已写入 {out_md} 与 {out_json}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
