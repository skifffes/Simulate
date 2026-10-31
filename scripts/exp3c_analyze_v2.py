import csv,glob,json,math,numpy as np
from pathlib import Path
ROOT=Path('/scratch/lijh/simulate_vla');R=ROOT/'results/experiment_3_controlled';D=R/'data';S=R/'summaries';S.mkdir(parents=True,exist_ok=True)
def quat(q):return np.array(q,float)
def qmul(a,b):
 w1,x1,y1,z1=a;w2,x2,y2,z2=b
 return np.array([w1*w2-x1*x2-y1*y2-z1*z2,w1*x2+x1*w2+y1*z2-z1*y2,w1*y2-x1*z2+y1*w2+z1*x2,w1*z2+x1*y2-y1*x2+z1*w2])
def qconj(q):return np.array([q[0],-q[1],-q[2],-q[3]])
def qrot(q,v):
 qv=np.array([0,*v]);return qmul(qmul(q,qv),qconj(q))[1:]
def qangle(a,b):
 d=abs(float(np.dot(a,b)));return math.degrees(2*math.acos(min(1,d)))
def yaw_axis(q):
 x,y,z,w=q;return math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
def axis_angle(a,b):
 d=abs(math.atan2(math.sin(yaw_axis(a)-yaw_axis(b)),math.cos(yaw_axis(a)-yaw_axis(b))));return math.degrees(min(d,math.pi-d))
def rel(ee,eeq,obj,objq):
 # T_EE^-1 T_object
 p=qrot(qconj(eeq),np.asarray(obj)-np.asarray(ee));q=qmul(qconj(eeq),objq);return p,q
rows=[];acts=[];align=[]
for f in sorted(glob.glob(str(R/'rollout_v2/*/summary.json'))):
 p=Path(f);tag=p.parent.name;parts=tag.split('_');g=parts[0];mode=parts[1];seed=int(parts[2].lstrip('r'))
 d=json.load(open(f));init=json.load(open(p.parent/'initial_state.json'))
 ph=list(csv.DictReader(open(p.parent/'physics_trace.csv',encoding='utf-8-sig')))[-1]
 first=None
 for c in csv.DictReader(open(p.parent/'contact_trace.csv',encoding='utf-8-sig')):
  if c.get('bodyA')=='1' or c.get('bodyB')=='1':first=c;break
 pol=[json.loads(x) for x in open(p.parent/'policy_trace.jsonl',encoding='utf-8') if x.strip()]
 a0=pol[0]['action'] if pol else [float('nan')]*8
 rows.append({'group':g,'mode':mode,'seed':seed,'success_v22':d['success'],'wall_contact':d['wall_contact'],'max_lift_mm':float(d['max_lift_m'])*1000,
  'gap_mm':float(ph['obj_wall_gap_m'])*1000,'dy_mm':float(ph['obj_dy_m'])*1000,'tilt_deg':float(ph['obj_tilt_deg']),'n_policy_steps':len(pol),
  'first_contact':(first['nameA'] if first and first['bodyA']=='1' else first['nameB']) if first else '','first_contact_t':float(first['physics_time_sec']) if first else ''})
 acts.append({'group':g,'mode':mode,'seed':seed,**{f'q{i+1}':float(a0[i]) for i in range(7)},'grip_mm':float(a0[7])*1000})
with open(D/'b90_rollout_results_v2.csv','w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
with open(D/'first_action_v2.csv','w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(acts[0].keys()));w.writeheader();w.writerows(acts)
# actual startup records
irec={}
for f in glob.glob(str(R/'rollout_v2/*/initial_state.json')):
 p=Path(f);tag=p.parent.name;parts=tag.split('_');g=parts[0];mode=parts[1];seed=int(parts[2].lstrip('r'))
 d=json.load(open(f));z=np.load(p.parent/'initial_observation.npz');irec[(g,mode,seed)]=(d,z)
# alignment per seed for B/C and A/B direct
metrics=[]
for seed in range(5):
 for mode in ('snapshot','continuous'):
  pairs=[('A','B'),('B','C'),('A','C')] if mode=='snapshot' else [('B','C')]
  for a,b in pairs:
   if (a,mode,seed) not in irec or (b,mode,seed) not in irec:continue
   A,az=irec[(a,mode,seed)];B,bz=irec[(b,mode,seed)]
   pa,qa=rel(A['ee'],A['ee_quat'],A['obj_pos'],A['obj_quat']);pb,qb=rel(B['ee'],B['ee_quat'],B['obj_pos'],B['obj_quat'])
   metrics.append({'seed':seed,'pair':f'{a}_vs_{b}','mode':mode,
    'q_l2_rad':float(np.linalg.norm(np.array(A['q'])-np.array(B['q']))),'q_max_rad':float(np.max(np.abs(np.array(A['q'])-np.array(B['q'])))),
    'dq_l2':float(np.linalg.norm(np.array(A['dq'])-np.array(B['dq']))),'ee_pos_mm':float(np.linalg.norm(np.array(A['ee'])-np.array(B['ee']))*1000),
    'ee_rot_deg':qangle(quat(A['ee_quat']),quat(B['ee_quat'])),'grip_mm':abs(float(A['grip_target'])-float(B['grip_target']))*1000,
    'obj_pos_mm':float(np.linalg.norm(np.array(A['obj_pos'])-np.array(B['obj_pos']))*1000),
    'obj_axis_deg':axis_angle(A['obj_quat'],B['obj_quat']),
    'rel_pos_mm':float(np.linalg.norm(pa-pb)*1000),'rel_rot_deg':qangle(qa,qb),
    'top_mae':float(np.mean(np.abs(az['top'].astype(float)-bz['top'].astype(float)))),
    'wrist_mae':float(np.mean(np.abs(az['wrist'].astype(float)-bz['wrist'].astype(float))))})
with open(D/'state_alignment_v2.csv','w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(metrics[0].keys()));w.writeheader();w.writerows(metrics)
# summaries
grp=[]
for mode in ('snapshot','continuous'):
 for g in ('A','B','C'):
  s=[x for x in rows if x['mode']==mode and x['group']==g]
  if not s:continue
  grp.append({'mode':mode,'group':g,'n':len(s),'success':sum(bool(x['success_v22']) for x in s),'wall':sum(bool(x['wall_contact']) for x in s),
   'mean_gap_mm':float(np.mean([x['gap_mm'] for x in s])),'mean_tilt_deg':float(np.mean([x['tilt_deg'] for x in s]))})
with open(S/'group_summary_v2.csv','w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(grp[0].keys()));w.writeheader();w.writerows(grp)
# full 7-joint action L2
fa={(x['group'],x['mode'],x['seed']):np.array([x[f'q{i}'] for i in range(1,8)],float) for x in acts}
fd=[]
for seed in range(5):
 for a,b in [('A','B'),('B','C'),('A','C')]:
  fd.append({'seed':seed,'pair':f'{a}_vs_{b}','q_l2_7':float(np.linalg.norm(fa[(a,'snapshot',seed)]-fa[(b,'snapshot',seed)]))})
for seed in range(5):
 fd.append({'seed':seed,'pair':'B_vs_C_continuous','q_l2_7':float(np.linalg.norm(fa[('B','continuous',seed)]-fa[('C','continuous',seed)]))})
with open(S/'first_action_diff_v2.csv','w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(fd[0].keys()));w.writeheader();w.writerows(fd)
print('ok',len(rows),len(metrics))
