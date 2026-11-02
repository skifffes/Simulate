"""生成实验 4.1 的场景/目标注册表（含 pilot 可达性统计）。

输入：
  * `src/exp41/scene41.py` 的 A/B 目标、B 物件起点、B 机器人起点、相机版本与 SHA256；
  * pilot 的 `attempts.csv`（按 goal/config 汇总真实物理成功率）。
输出：`results/experiment_4_1/scene_space/goal_registry.json` 与 `b_start_states.json`。

用法：python scripts/exp41/make_registry.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / 'src', ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from exp41 import scene41  # noqa: E402


def main() -> int:
    out = ROOT / 'results' / 'experiment_4_1' / 'scene_space'
    out.mkdir(parents=True, exist_ok=True)
    reg = scene41.registry()

    attempts = []
    ap = ROOT / 'results' / 'experiment_4_1' / 'pilot' / 'attempts.csv'
    if ap.exists():
        attempts = list(csv.DictReader(open(ap, encoding='utf-8')))

    def summarize(kind: str, key: str) -> dict:
        stats = {}
        rows = [r for r in attempts if r['skill'] == kind]
        for r in rows:
            k = r[key]
            d = stats.setdefault(k, {'attempts': 0, 'success': 0, 'failure_reasons': {}})
            d['attempts'] += 1
            if r['strict_success'].lower() == 'true':
                d['success'] += 1
            else:
                fr = r['failure_reason'] or 'unknown'
                d['failure_reasons'][fr] = d['failure_reasons'].get(fr, 0) + 1
        for k, d in stats.items():
            d['success_rate'] = round(d['success'] / max(1, d['attempts']), 3)
        return stats

    a_stats = summarize('A90', 'goal_id')
    b_stats = summarize('B90', 'config_id')

    for g in reg['a_goals']:
        s = a_stats.get(g['goal_id'], {})
        g['pilot_attempts'] = s.get('attempts', 0)
        g['pilot_success'] = s.get('success', 0)
        g['pilot_success_rate'] = s.get('success_rate', None)
    for g in reg['b_goals']:
        g['aligned_configs'] = [c for c in scene41.B_ALIGNED_PAIRS if c[1] == g['goal_id']]
    for o in reg['b_object_starts']:
        o['aligned_goal'] = next((g for obj, g in scene41.B_ALIGNED_PAIRS
                                  if obj == o['start_id']), None)
    for r in reg['b_robot_starts']:
        r['pilot'] = {c: v for c, v in b_stats.items() if c.endswith(r['robot_start_id'])}

    reg['pilot_summary'] = {'A90_by_goal': a_stats, 'B90_by_config': b_stats,
                            'source': 'results/experiment_4_1/pilot/attempts.csv'}
    reg['notes'] = [
        'A 的 9 个目标是预注册网格（±3 cm），pilot 里每个目标都跑过真实抓取/放置，未按结果筛选。',
        f'每条 A episode 的目标点 = 目标中心 + 确定性抖动（±{scene41.A_GOAL_JITTER_M * 1000:.0f} mm，'
        '由种子决定，仍在 2 cm 容差内）；指令文本写出实际目标点，评价也按该点计算。'
        '这样 A 终态多样性来自不同目标点，而不是同一点重复采样。',
        'B 的主数据只使用 (物体起点, 目标区) 预先配对的 3 组；跨区配对需要横向推动，'
        '当前专家在该情形不稳定（见 pilot 里的跨区失败记录），因此不进入主数据，'
        '也不声称已实现"同一起点按目标切换动作"的条件泛化。',
        '相机为实验 4.0 冻结版本 exp4.0-camera-v3，SHA256 见 camera_sha256。',
    ]
    (out / 'goal_registry.json').write_text(json.dumps(reg, ensure_ascii=False, indent=2),
                                            encoding='utf-8')
    (out / 'b_start_states.json').write_text(json.dumps({
        'object_starts': reg['b_object_starts'], 'robot_starts': reg['b_robot_starts'],
        'aligned_pairs': scene41.B_ALIGNED_PAIRS,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"wrote {out/'goal_registry.json'}  A goals={len(reg['a_goals'])} B goals={len(reg['b_goals'])}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
