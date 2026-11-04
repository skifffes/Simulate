import json
import sys

d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else '/tmp/picks_v2.json'))
print(json.dumps(d.get('stats', {}), ensure_ascii=False, indent=2))
print('phases:', sorted({i['phase'] for i in d['items']}))
print('n_items:', len(d['items']))
