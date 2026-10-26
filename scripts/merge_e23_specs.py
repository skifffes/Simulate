"""实验 2.3：把 D1 的 yaw0/yaw90 spec 合并成 D2 使用的 120 条列表。"""
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1] / 'data' / '_shards_e23'
yaw0 = json.loads((root / 'used_yaw0.json').read_text(encoding='utf-8'))
yaw90 = json.loads((root / 'used_yaw90.json').read_text(encoding='utf-8'))
specs = yaw0 + yaw90
(root / 'used_specs.json').write_text(json.dumps(specs, ensure_ascii=False, indent=2),
                                     encoding='utf-8')
print(json.dumps({'yaw0': len(yaw0), 'yaw90': len(yaw90), 'total': len(specs),
                  'gap_yaw0': [min(s['gap_target_m'] for s in yaw0),
                               max(s['gap_target_m'] for s in yaw0)],
                  'gap_yaw90': [min(s['gap_target_m'] for s in yaw90),
                                max(s['gap_target_m'] for s in yaw90)]}, indent=2))
