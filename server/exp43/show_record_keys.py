import glob
import json

f = sorted(glob.glob('/scratch/lijh/exp43/a90a/records/*.json'))[0]
r = json.load(open(f))
print('A90 record keys:', sorted(r.keys()))
print('A90 metrics keys:', sorted((r.get('metrics') or {}).keys()))
g = sorted(glob.glob('/scratch/lijh/exp43/b90a/records/*.json'))[0]
q = json.load(open(g))
print('B90 record keys:', sorted(q.keys()))
print('B90 metrics keys:', sorted((q.get('metrics') or {}).keys()))
print('A90 metrics sample:', json.dumps(r.get('metrics'), ensure_ascii=False)[:400])
print('B90 metrics sample:', json.dumps(q.get('metrics'), ensure_ascii=False)[:400])
