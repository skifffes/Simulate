import csv,statistics
from pathlib import Path
R=Path(r'D:\python\simulate\results\experiment_3_controlled');D=R/'data';S=R/'summaries';S.mkdir(exist_ok=True)
r=list(csv.DictReader((D/'b90_rollout_results.csv').open(encoding='utf-8-sig')))
rows=[]
for g in 'ABC':
 s=[x for x in r if x['condition']==g]
 rows.append({'group':g,'n_rollouts':len(s),'v22_success':sum(x['success_v22']=='True' for x in s),'wall_contact':sum(x['wall_contact']=='True' for x in s),
  'success_rate':sum(x['success_v22']=='True' for x in s)/len(s),'mean_tilt_deg':statistics.mean(float(x['final_tilt_deg']) for x in s),
  'mean_gap_mm':statistics.mean(float(x['final_gap_mm']) for x in s),'mean_lift_mm':statistics.mean(float(x['max_lift_mm']) for x in s)})
with (S/'group_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
seedrows=[]
for s in '01234':
 d={x['condition']:x for x in r if x['seed']==s}
 seedrows.append({'seed':s,'A_success':d['A']['success_v22'],'B_success':d['B']['success_v22'],'C_success':d['C']['success_v22'],
  'A_tilt':d['A']['final_tilt_deg'],'B_tilt':d['B']['final_tilt_deg'],'C_tilt':d['C']['final_tilt_deg'],
  'A_gap':d['A']['final_gap_mm'],'B_gap':d['B']['final_gap_mm'],'C_gap':d['C']['final_gap_mm'],
  'A_wall':d['A']['wall_contact'],'B_wall':d['B']['wall_contact'],'C_wall':d['C']['wall_contact']})
with (S/'paired_seed.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(seedrows[0].keys()));w.writeheader();w.writerows(seedrows)
fa=list(csv.DictReader((D/'first_action.csv').open(encoding='utf-8-sig')));d={(x['condition'],x['seed']):x for x in fa}
diff=[]
for s in '01234':
 def vec(g):x=d[(g,s)];return [float(x['a0_q1']),float(x['a0_q4']),float(x['a0_q7'])]
 import math
 ab=math.dist(vec('A'),vec('B'));bc=math.dist(vec('B'),vec('C'));ac=math.dist(vec('A'),vec('C'))
 diff.append({'seed':s,'A_B_q_l2_rad':ab,'B_C_q_l2_rad':bc,'A_C_q_l2_rad':ac})
with (S/'first_action_diff.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(diff[0].keys()));w.writeheader();w.writerows(diff)
print('ok')
