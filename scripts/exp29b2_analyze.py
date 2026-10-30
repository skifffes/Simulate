import csv, glob, json, math, statistics
from pathlib import Path
import numpy as np
ROOT=Path('/scratch/lijh/simulate_vla'); BASE=ROOT/'results/experiment_2_9/b2_low_near'
OUT=BASE/'data'; OUT.mkdir(parents=True,exist_ok=True)

def parse(tag):
 p=tag.split('_g'); sid=p[0]; g=p[1].split('_r'); return sid,int(g[0]),int(g[1])

rows=[]; traj=[]; ctrl=[]; contacts=[]
for f in sorted((BASE/'runs').glob('*/*/*/*/summary.json')):
 d=f.parent; rel=d.relative_to(BASE/'runs'); kind=rel.parts[0]; tag=rel.parts[1]; sid,w,rep=parse(tag)
 s=json.loads(f.read_text()); ph=list(csv.DictReader((d/'physics_trace.csv').open(encoding='utf-8-sig'))); last=ph[-1]
 first=None
 for c in csv.DictReader((d/'contact_trace.csv').open(encoding='utf-8-sig')):
  if c.get('bodyA')=='1' or c.get('bodyB')=='1': first=c;break
 ee=''
 if first:
  ct=float(first['physics_time_sec']); rr=min(ph,key=lambda x:abs(float(x['physics_time_sec'])-ct));ee=float(rr['ee_z'])
  contacts.append({'run_id':f'{kind}_{tag}','condition':kind,'state_id':sid,'width_mm':w,'rep':rep,
                   'time_s':ct,'link':first['nameA'] if first['bodyA']=='1' else first['nameB'],
                   'normal_x':float(first['normal_x']),'normal_z':float(first['normal_z']),
                   'ee_z_mm':ee*1000})
 row={'run_id':f'{kind}_{tag}','condition':kind,'state_id':sid,'width_mm':w,'rep':rep,
      'success':bool(s['success']),'max_lift_mm':float(s['max_lift_m'])*1000,
      'wall_contact':bool(s['wall_contact']),'final_gap_mm':float(last['obj_wall_gap_m'])*1000,
      'final_dy_mm':float(last['obj_dy_m'])*1000,'final_tilt_deg':float(last['obj_tilt_deg']),
      'first_link':(first['nameA'] if first and first['bodyA']=='1' else first['nameB']) if first else '',
      'first_time_s':float(first['physics_time_sec']) if first else '',
      'contact_ee_z_mm':ee*1000 if ee!='' else '',
      'path_len_m':0.0}
 rows.append(row)
 xs=np.array([float(r['ee_x']) for r in ph]);zs=np.array([float(r['ee_z']) for r in ph])
 row['path_len_m']=float(np.sum(np.sqrt(np.diff(xs)**2+np.diff(zs)**2)))
 for k in range(0,len(ph),12):
  r=ph[k];traj.append({'run_id':f'{kind}_{tag}','condition':kind,'state_id':sid,'width_mm':w,'rep':rep,
                       't':r['physics_time_sec'],'ee_x':r['ee_x'],'ee_y':r['ee_y'],'ee_z':r['ee_z'],
                       'grip':r['grip_actual'],'obj_tilt':r['obj_tilt_deg'],'gap':r['obj_wall_gap_m']})
 pdir=d/'policy_trace.jsonl'
 if pdir.exists():
  for ln in pdir.read_text().splitlines():
   if not ln.strip():continue
   x=json.loads(ln)
   if x.get('skill')=='B':
    ctrl.append({'run_id':f'{kind}_{tag}','condition':kind,'state_id':sid,'width_mm':w,'rep':rep,
                 'policy_step':x['policy_step'],'time_s':x['physics_time_sec'],
                 **{f'a{j+1}':x['action'][j] for j in range(7)},'a_grip':x['action'][7]})
with (OUT/'rollout_outcomes.csv').open('w',newline='',encoding='utf-8-sig') as f:
 wri=csv.DictWriter(f,fieldnames=list(rows[0].keys()));wri.writeheader();wri.writerows(rows)
with (OUT/'actual_ee_trajectories.csv').open('w',newline='',encoding='utf-8-sig') as f:
 wri=csv.DictWriter(f,fieldnames=list(traj[0].keys()));wri.writeheader();wri.writerows(traj)
with (OUT/'contact_events.csv').open('w',newline='',encoding='utf-8-sig') as f:
 wri=csv.DictWriter(f,fieldnames=list(contacts[0].keys()));wri.writeheader();wri.writerows(contacts)
with (OUT/'controller_targets.csv').open('w',newline='',encoding='utf-8-sig') as f:
 wri=csv.DictWriter(f,fieldnames=list(ctrl[0].keys()));wri.writeheader();wri.writerows(ctrl)
with (OUT/'expert_trajectories.csv').open('w',newline='',encoding='utf-8-sig') as f:
 wri=csv.DictWriter(f,fieldnames=list(traj[0].keys()));wri.writeheader();wri.writerows([r for r in traj if r['condition'] in ('E0','E1')])
with (OUT/'policy_query_events.csv').open('w',newline='',encoding='utf-8-sig') as f:
 wri=csv.writer(f);wri.writerow(['run_id','query_interval_policy_ticks','query_timestamps','note'])
 for r in rows:
  if r['condition']=='D2':wri.writerow([r['run_id'],50,'not_logged','n_action_steps=50 inferred; exact query times not logged'])
# pair comparison 14 vs35/40 within same state/rep
pairs=[]
for sid in sorted(set(r['state_id'] for r in rows)):
 for w in (35,40):
  for rep in (0,1):
   a=next((r for r in rows if r['condition']=='D2' and r['state_id']==sid and r['width_mm']==14 and r['rep']==rep),None)
   b=next((r for r in rows if r['condition']=='D2' and r['state_id']==sid and r['width_mm']==w and r['rep']==rep),None)
   if not a or not b:continue
   pairs.append({'state_id':sid,'width_mm':w,'rep':rep,
                 'success_14':int(a['success']),'success_wide':int(b['success']),
                 'contact_z_14_mm':a['contact_ee_z_mm'],'contact_z_wide_mm':b['contact_ee_z_mm'],
                 'contact_t_14_s':a['first_time_s'],'contact_t_wide_s':b['first_time_s'],
                 'lift_14_mm':a['max_lift_mm'],'lift_wide_mm':b['max_lift_mm'],
                 'gap_14_mm':a['final_gap_mm'],'gap_wide_mm':b['final_gap_mm']})
with (OUT/'trajectory_pair_comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
 wri=csv.DictWriter(f,fieldnames=list(pairs[0].keys()));wri.writeheader();wri.writerows(pairs)
print(len(rows),len(pairs))
