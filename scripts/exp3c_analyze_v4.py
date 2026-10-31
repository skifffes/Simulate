import csv,glob,json,math,hashlib,numpy as np
from pathlib import Path
ROOT=Path('/scratch/lijh/simulate_vla');R=ROOT/'results/experiment_3_controlled';D=R/'data';S=R/'summaries'
B=next(x for x in csv.DictReader(open(ROOT/'results/experiment_3/data/b_training_start_states.csv',encoding='utf-8-sig')) if x['b_candidate_id']=='ep064_f000')
qb=np.array([float(x) for x in B['q'].split(';')])
def qmul(a,b):
    ax,ay,az,aw=a;bx,by,bz,bw=b
    return np.array([aw*bx+ax*bw+ay*bz-az*by,
                     aw*by-ax*bz+ay*bw+az*bx,
                     aw*bz+ax*by-ay*bx+az*bw,
                     aw*bw-ax*bx-ay*by-az*bz])
def qconj(q):return np.array([-q[0],-q[1],-q[2],q[3]])
def qrot(q,v):return qmul(qmul(q,np.array([*v,0.0])),qconj(q))[:3]
def qang(a,b):d=abs(float(np.dot(a,b)));return math.degrees(2*math.acos(min(1,d)))
def rel(ee,eeq,obj,objq):return qrot(qconj(np.array(eeq)),np.array(obj)-np.array(ee)),qmul(qconj(np.array(eeq)),np.array(objq))
rows=[];fa=[];align=[]
for f in sorted(glob.glob(str(R/'rollout_v4/*/summary.json'))):
 p=Path(f);tag=p.parent.name;g=tag[0];seed=int(tag.split('_')[-1].lstrip('r'));d=json.load(open(f));init=json.load(open(p.parent/'initial_state.json'))
 ph=list(csv.DictReader(open(p.parent/'physics_trace.csv',encoding='utf-8-sig')))[-1]
 pol=[json.loads(x) for x in open(p.parent/'policy_trace.jsonl',encoding='utf-8') if x.strip()]
 raw=np.load(p.parent/'raw_actions.npz')['actions'];a0=raw[0]
 contacts=list(csv.DictReader(open(p.parent/'contact_trace.csv',encoding='utf-8-sig')))
 sfile=R/f'state_audit_v4/{g}_continuous_r{seed}/initial_state.json'
 audit=json.load(open(sfile)) if sfile.exists() else {}
 if audit:
  cs=audit.get('contacts_start',{});cc={'robot_object':cs.get('robot_object',0),'object_table':cs.get('object_table',0),'object_wall':cs.get('object_wall',0),'other':cs.get('robot_other',0)};n0=sum(cc.values())
 else:
  first_t=float(pol[0]['physics_time_sec']) if pol else 0.0
  win=[c for c in contacts if first_t<=float(c['physics_time_sec'])<=first_t+0.01]
  def cat(c):
   a,b=c['nameA'].lower(),c['nameB'].lower();pair={a,b}
   if any('robot' in x for x in pair) and any('block' in x for x in pair):return 'robot_object'
   if any('block' in x for x in pair) and any('wall' in x for x in pair):return 'object_wall'
   if any('block' in x for x in pair) and any('table' in x for x in pair):return 'object_table'
   return 'other'
  cc={'robot_object':0,'object_table':0,'object_wall':0,'other':0}
  for c in win:cc[cat(c)]+=1
  n0=sum(cc.values())
 rows.append({'group':g,'seed':seed,'success_v22':d['success'],'wall_contact':d['wall_contact'],'max_lift_mm':float(d['max_lift_m'])*1000,'gap_mm':float(ph['obj_wall_gap_m'])*1000,'dy_mm':float(ph['obj_dy_m'])*1000,'tilt_deg':float(ph['obj_tilt_deg']),'n_policy_steps':len(pol),'converged':init.get('converged',False),'conv_steps':init.get('conv_steps',-1),'q_err_to_qb':float(np.linalg.norm(np.array(init['q'])-qb)),'dq_norm':float(np.linalg.norm(init['dq'])),'n_contacts_start':n0,'robot_object_contacts':cc['robot_object'],'object_table_contacts':cc['object_table'],'object_wall_contacts':cc['object_wall'],'other_contacts':cc['other']})
 fa.append({'group':g,'seed':seed,**{f'q{i+1}':float(a0[i]) for i in range(7)},'grip_mm':float(a0[7])*1000})
with open(D/'b90_rollout_results_v4.csv','w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
with open(D/'first_action_v4.csv','w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(fa[0].keys()));w.writeheader();w.writerows(fa)
for seed in range(5):
 ai=json.load(open(R/f'rollout_v4/B_continuous_r{seed}/initial_state.json'));ci=json.load(open(R/f'rollout_v4/C_continuous_r{seed}/initial_state.json'))
 ab=json.load(open(R/f'state_audit_v4/B_continuous_r{seed}/initial_state.json'));cb=json.load(open(R/f'state_audit_v4/C_continuous_r{seed}/initial_state.json'))
 cs=ab.get('contacts_start',{})
 az=np.load(R/f'rollout_v4/B_continuous_r{seed}/model_input.npz');cz=np.load(R/f'rollout_v4/C_continuous_r{seed}/model_input.npz')
 pa,qa=rel(ai['ee'],ai['ee_quat'],ai['obj_pos'],ai['obj_quat']);pc,qc=rel(ci['ee'],ci['ee_quat'],ci['obj_pos'],ci['obj_quat'])
 actB=np.array([fa[i][f'q{j}'] for i in range(len(fa)) if fa[i]['group']=='B' and fa[i]['seed']==seed for j in range(1,8)],float)
 actC=np.array([fa[i][f'q{j}'] for i in range(len(fa)) if fa[i]['group']=='C' and fa[i]['seed']==seed for j in range(1,8)],float)
 align.append({'seed':seed,'q_l2_rad':float(np.linalg.norm(np.array(ai['q'])-np.array(ci['q']))),'dq_l2':float(np.linalg.norm(np.array(ai['dq'])-np.array(ci['dq']))),'ee_pos_mm':float(np.linalg.norm(np.array(ai['ee'])-np.array(ci['ee']))*1000),'obj_pos_mm':float(np.linalg.norm(np.array(ai['obj_pos'])-np.array(ci['obj_pos']))*1000),'obj_vel_l2':float(np.linalg.norm(np.array(ai['obj_vel'])-np.array(ci['obj_vel']))),'rel_pos_mm':float(np.linalg.norm(pa-pc)*1000),'rel_rot_deg':qang(qa,qc),'grip_left_mm':abs(ai['grip_actual'][0]-ci['grip_actual'][0])*1000,'grip_right_mm':abs(ai['grip_actual'][1]-ci['grip_actual'][1])*1000,'top_input_mae':float(np.mean(np.abs(az['top'].astype(float)-cz['top'].astype(float)))),'wrist_input_mae':float(np.mean(np.abs(az['wrist'].astype(float)-cz['wrist'].astype(float)))),'state_input_l2':float(np.linalg.norm(az['state'].astype(float)-cz['state'].astype(float))),'action7_l2':float(np.linalg.norm(actB-actC)),'top_hash_equal':hashlib.sha256(az['top'].tobytes()).hexdigest()==hashlib.sha256(cz['top'].tobytes()).hexdigest(),'q_err_to_qb':float(np.linalg.norm(np.array(ai['q'])-qb)),'dq_norm':float(np.linalg.norm(ai['dq'])),'grip_actual_left':float(ai['grip_actual'][0]),'grip_actual_right':float(ai['grip_actual'][1]),'finger_dq_l2':float(np.linalg.norm(np.array(ab['finger_dq'])-np.array(cb['finger_dq']))),'finger_dq_left':float(ab['finger_dq'][0]),'finger_dq_right':float(ab['finger_dq'][1]),'robot_object_contacts':cs.get('robot_object',0),'object_table_contacts':cs.get('object_table',0),'object_wall_contacts':cs.get('object_wall',0),'n_contacts_start':sum(cs.values())})
with open(D/'state_alignment_v4.csv','w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(align[0].keys()));w.writeheader();w.writerows(align)
grp=[]
for g in 'BC':
 s=[x for x in rows if x['group']==g];grp.append({'group':g,'n':len(s),'success':sum(bool(x['success_v22']) for x in s),'wall':sum(bool(x['wall_contact']) for x in s),'mean_gap_mm':float(np.mean([x['gap_mm'] for x in s])),'mean_tilt_deg':float(np.mean([x['tilt_deg'] for x in s])),'max_q_err_rad':max(x['q_err_to_qb'] for x in s),'max_dq_norm':max(x['dq_norm'] for x in s),'max_contacts_start':max(x['n_contacts_start'] for x in s)})
with open(S/'group_summary_v4.csv','w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(grp[0].keys()));w.writeheader();w.writerows(grp)
print('ok')
