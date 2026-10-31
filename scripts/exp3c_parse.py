import csv,glob,json,math,numpy as np
from pathlib import Path
R=Path('/scratch/lijh/simulate_vla/results/experiment_3_controlled');rows=[];first=[]
for f in sorted(glob.glob(str(R/'rollout/*/*/*/summary.json'))):
 d=json.load(open(f));p=Path(f);tag=p.parent.parent.parent.name;g=tag.split('_')[0];seed=int(tag.split('_')[-1].lstrip('r'));kind=p.parent.name
 ph=list(csv.DictReader(open(p.parent/'physics_trace.csv',encoding='utf-8-sig')))[-1]
 firstc=None
 for c in csv.DictReader(open(p.parent/'contact_trace.csv',encoding='utf-8-sig')):
  ba,bb=c.get('bodyA',''),c.get('bodyB','')
  if ba=='1' or bb=='1':
   firstc=c;break
 pol=[]
 for line in open(p.parent/'policy_trace.jsonl',encoding='utf-8'):
  if line.strip():pol.append(json.loads(line))
 act0=pol[0]['action'] if pol else None
 init=json.load(open(p.parent/'initial_state.json'))
 rows.append({'condition':g,'seed':seed,'tag':tag,'success_v22':bool(d['success']),'wall_contact':bool(d['wall_contact']),
  'max_lift_mm':float(d['max_lift_m'])*1000,'final_gap_mm':float(ph['obj_wall_gap_m'])*1000,'final_dy_mm':float(ph['obj_dy_m'])*1000,
  'final_tilt_deg':float(ph['obj_tilt_deg']),'first_contact_name':(firstc['nameA'] if firstc and firstc['bodyA']=='1' else firstc['nameB']) if firstc else '',
  'first_contact_t':float(firstc['physics_time_sec']) if firstc else '', 'n_policy_steps':len(pol),'initial_q_l2':''})
 if act0:first.append({'condition':g,'seed':seed,'a0_q1':act0[0],'a0_q4':act0[3],'a0_q7':act0[6],'a0_grip_mm':act0[7]*1000})
with open(R/'data/b90_rollout_results.csv','w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
with open(R/'data/first_action.csv','w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(first[0].keys()));w.writeheader();w.writerows(first)
print('ok',len(rows))
