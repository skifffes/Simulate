"""统计 pilot 记录里的专家失败原因（诊断用）。"""
import collections
import glob
import json
import sys

d = sys.argv[1] if len(sys.argv) > 1 else '.'
rows = []
for f in sorted(glob.glob(d + '/*.json')):
    r = json.load(open(f, encoding='utf-8'))
    rows.append(r)
print('n', len(rows))
c = collections.Counter((r['goal_id'], r['expert_ok'], r['expert_reason']) for r in rows)
for k, v in sorted(c.items()):
    print(k, v)
print('--- successful-by-eval configs ---')
c2 = collections.Counter((r['goal_id'], r.get('config_id'), r['strict_success']) for r in rows)
for k, v in sorted(c2.items()):
    print(k, v)
print('--- sample failing record ---')
for r in rows:
    if not r['strict_success'] and r['goal_id'].endswith('mid'):
        print(json.dumps({k: r[k] for k in ('goal_id', 'config_id', 'expert_ok', 'expert_reason',
                                            'failure_reason', 'expert_metrics')},
                         ensure_ascii=False, indent=1)[:1400])
        break
