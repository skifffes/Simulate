import csv,glob,json,math,hashlib,numpy as np
from pathlib import Path
ROOT=Path('/scratch/lijh/simulate_vla');R=ROOT/'results/experiment_3_2';D=R/'data';S=R/'summaries';D.mkdir(exist_ok=True);S.mkdir(exist_ok=True)
A=next(csv.DictReader(open(ROOT/'results/experiment_3/data/a_terminal_states.csv',encoding='utf-8-sig')))
BROW={x['b_candidate_id']:x for x in csv.DictReader(open(ROOT/'results/experiment_3/data/b_training_start_states.csv',encoding='utf-8-sig'))}
def qmul(a,b):
 ax,ay,az,aw=a;bx,by,bz,bw=b
 return np.array([aw*bx+ax*bw+ay*bz-az*by,aw*by-ax*bz+ay*bw+az*bx,aw*bz+ax*by-ay*bx+az*bw,aw*bw-ax*bx-ay*by-az*bz])
def qconj(q):return np.array([-q[0],-q[1],-q[2],q[3]])
def qrot(q,v):return qmul(qmul(q,np.array([*v,0])),qconj(q))[:3]
def qang(a,b):return math.degrees(2*math.acos(min(1,abs(float(np.dot(a,b))))))
def rel(ee,eeq,obj,objq):return qrot(qconj(eeq),np.array(obj)-np.array(ee)),qmul(qconj(eeq),np.array(objq))
rows=[];acts=[];align=[]
for f in sorted(glob.glob(str(R/'rollout/*/summary.json'))):
 p=Path(f);tag=p.parent.name;parts=tag.split('_');g=parts[0];target=parts[1]+'_'+parts[2];seed=int(parts[3][1:]);d=json.load(open(f));init=json.load(open(p.parent/'initial_state.json'))
 ph=list(csv.DictReader(open(p.parent/'physics_trace.csv',encoding='utf-8-sig')))[-1]
 pol=[json.loads(x) for x in open(p.parent/'policy_trace.jsonl',encoding='utf-8') if x.strip()]
 raw=np.load(p.parent/'raw_actions.npz')['actions'];a0=raw[0]
 rows.append({'group':g,'target':target,'seed':seed,'success_v22':d['success'],'wall_contact':d['wall_contact'],'max_lift_mm':float(d['max_lift_m'])*1000,'gap_mm':float(ph['obj_wall_gap_m'])*1000,'dy_mm':float(ph['obj_dy_m'])*1000,'tilt_deg':float(ph['obj_tilt_deg']),'n_policy_steps':len(pol),'converged':init['converged'],'conv_steps':init['conv_steps'],'q_err_to_target':float(np.linalg.norm(np.array(init['q'])-np.array(init['q_target']))),'dq_norm':float(np.linalg.norm(init['dq'])),'robot_object_contacts':init['contacts_start']['robot_object'],'object_table_contacts':init['contacts_start']['object_table'],'object_wall_contacts':init['contacts_start']['object_wall']})
 acts.append({'group':g,'target':target,'seed':seed,**{f'q{i+1}':float(a0[i]) for i in range(7)},'grip_mm':float(a0[7])*1000})
with open(D/'rollout_results.csv','w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
with open(D/'first_action_comparison.csv','w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(acts[0].keys()));w.writeheader();w.writerows(acts)
# alignment per target/seed B/C and A/B
for target in sorted(set(x['target'] for x in rows)):
 for seed in range(5):
  ai=json.load(open(R/f'rollout/B_{target}_r{seed}/initial_state.json'));bi=json.load(open(R/f'rollout/C_{target}_r{seed}/initial_state.json'));aa=json.load(open(R/f'rollout/A_{target}_r{seed}/initial_state.json'))
  az=np.load(R/f'rollout/B_{target}_r{seed}/model_input.npz');bz=np.load(R/f'rollout/C_{target}_r{seed}/model_input.npz');aa_z=np.load(R/f'rollout/A_{target}_r{seed}/model_input.npz')
  pa,qqa=rel(aa['ee'],aa['ee_quat'],aa['obj_pos'],aa['obj_quat']);pb,qqb=rel(bi['ee'],bi['ee_quat'],bi['obj_pos'],bi['obj_quat']);pc,qqc=rel(ai['ee'],ai['ee_quat'],ai['obj_pos'],ai['obj_quat'])
  aB=np.array([acts[i][f'q{j}'] for i in range(len(acts)) if acts[i]['target']==target and acts[i]['group']=='B' and acts[i]['seed']==seed for j in range(1,8)],float)
  aC=np.array([acts[i][f'q{j}'] for i in range(len(acts)) if acts[i]['target']==target and acts[i]['group']=='C' and acts[i]['seed']==seed for j in range(1,8)],float)
  align.append({'target':target,'seed':seed,'BC_q_l2':float(np.linalg.norm(np.array(bi['q'])-np.array(ai['q']))),'BC_dq_l2':float(np.linalg.norm(np.array(bi['dq'])-np.array(ai['dq']))),'BC_ee_mm':float(np.linalg.norm(np.array(bi['ee'])-np.array(ai['ee']))*1000),'BC_rel_pos_mm':float(np.linalg.norm(pb-pc)*1000),'BC_rel_rot_deg':qang(qqb,qqc),'BC_grip_mm':float(np.linalg.norm(np.array(bi['grip_actual'])-np.array(ai['grip_actual']))*1000),'BC_finger_dq_l2':float(np.linalg.norm(np.array(bi['finger_dq'])-np.array(ai['finger_dq']))),'BC_state_input_l2':float(np.linalg.norm(az['state'].astype(float)-bz['state'].astype(float))),'BC_top_mae':float(np.mean(np.abs(az['top'].astype(float)-bz['top'].astype(float)))),'BC_wrist_mae':float(np.mean(np.abs(az['wrist'].astype(float)-bz['wrist'].astype(float)))),'BC_action7_l2':float(np.linalg.norm(aB-aC)),'BC_contacts_delta':int(sum(bi['contacts_start'].values())-sum(ai['contacts_start'].values())),'AB_obj_mm':float(np.linalg.norm(np.array(aa['obj_pos'])-np.array(bi['obj_pos']))*1000),'AB_rel_pos_mm':float(np.linalg.norm(pa-pb)*1000),'AB_rel_rot_deg':qang(qqa,qqb),'AB_state_input_l2':float(np.linalg.norm(aa_z['state'].astype(float)-az['state'].astype(float))),'AB_top_mae':float(np.mean(np.abs(aa_z['top'].astype(float)-az['top'].astype(float)))),'AB_action7_l2':float(np.linalg.norm(np.array([acts[i][f'q{j}'] for i in range(len(acts)) if acts[i]['target']==target and acts[i]['group']=='A' and acts[i]['seed']==seed for j in range(1,8)],float)-aB))})
with open(D/'state_alignment.csv','w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(align[0].keys()));w.writeheader();w.writerows(align)
# group summaries
grp=[]
for target in sorted(set(x['target'] for x in rows)):
 for g in 'ABC':
  s=[x for x in rows if x['target']==target and x['group']==g]
  grp.append({'target':target,'group':g,'n':len(s),'success':sum(bool(x['success_v22']) for x in s),'wall':sum(bool(x['wall_contact']) for x in s),'mean_gap_mm':float(np.mean([x['gap_mm'] for x in s])),'mean_tilt_deg':float(np.mean([x['tilt_deg'] for x in s])),'max_q_err':max(x['q_err_to_target'] for x in s)})
with open(S/'summary_by_target.csv','w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(grp[0].keys()));w.writeheader();w.writerows(grp)
print('ok',len(rows),len(align))
