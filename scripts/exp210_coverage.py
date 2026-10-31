import csv,json
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
ROOT=Path('/scratch/lijh/simulate_vla');OUT=ROOT/'results/experiment_2_10';DATA=OUT/'data'
B=pq.read_table(DATA/'b90_frame_state_action.parquet').to_pylist();H=pq.read_table(DATA/'expert_action_horizons.parquet').to_pylist()
Q=np.array([[r[f'q{j}'] for j in range(1,8)] for r in B]);G=np.array([r['gripper_per_finger_mm'] for r in B]);E=np.array([r['episode_id'] for r in B]);EZ=np.array([r['ee_z'] for r in B])
hmap={(r['episode_id'],r['frame_id'],r['horizon']):r['delta_z_mm'] for r in H}
states=[]
c3=list(csv.DictReader((ROOT/'results/experiment_2_6/C3_direct_B90/state_manifest.csv').open(encoding='utf-8-sig')))
for r in c3:
 states.append({'state_id':'C3_seed'+r['seed'],'source':'C3_terminal','q':[float(x) for x in r['q1..q7'].split(';')],
                'ee':[float(r['ee_x']),float(r['ee_y']),float(r['ee_z'])],'grip':float(r['grip_actual_m'])*1000})
s28=list(csv.DictReader((ROOT/'results/experiment_2_8/data/state_manifest.csv').open(encoding='utf-8-sig')))
for r in s28:
 states.append({'state_id':r['state_id'],'source':'2.8_B90_midstate','q':[float(x) for x in r['q'].split(';')],
                'ee':[float(r['ee_x']),float(r['ee_y']),float(r['ee_z'])],'grip':float(r['grip_obs_m'])*1000})
# ep66 f39 explicit
r=next(x for x in B if x['episode_id']==66 and x['frame_id']==39)
states.append({'state_id':'ep66_f39','source':'B90_training_frame','q':[r[f'q{j}'] for j in range(1,8)],
               'ee':[r['ee_x'],r['ee_y'],r['ee_z']],'grip':r['gripper_per_finger_mm']})
qrows=[];cov=[];neigh=[]
for s in states:
 q=np.array(s['q']);d=np.linalg.norm(Q-q,axis=1);order=np.argsort(d)
 qrows.append({'state_id':s['state_id'],'source':s['source'],'grip_mm':s['grip'],
               **{f'q{j+1}':float(q[j]) for j in range(7)},'ee_x':s['ee'][0],'ee_y':s['ee'][1],'ee_z':s['ee'][2],
               'nearest_q_rad':float(d.min()),'nearest_episode':int(E[order[0]])})
 for rad in (0.15,0.30,0.50,0.75):
  m=d<=rad;near=G[m];ee=E[m]
  c14=(near>=12.5)&(near<=16.5);c35=(near>=34.5)&(near<=35.5)
  m14=m & (G>=12.5) & (G<=16.5);m35=m & (G>=34.5) & (G<=35.5)
  cov.append({'state_id':s['state_id'],'radius_rad':rad,'n_frames':int(m.sum()),'n_episodes':int(len(set(ee.tolist()))),
              'narrow_frames':int(c14.sum()),'narrow_episodes':int(len(set(ee[c14].tolist()))),
              'wide_frames':int(c35.sum()),'wide_episodes':int(len(set(ee[c35].tolist()))),
              'common_support':bool(c14.sum()>=5 and c35.sum()>=5 and len(set(ee[c14].tolist()))>=3 and len(set(ee[c35].tolist()))>=3),
              'expert_dz_h0_narrow':float(np.mean([B[i]['dz_target_h0_mm'] for i in np.where(m14)[0]])) if m14.any() else '',
              'expert_dz_h0_wide':float(np.mean([B[i]['dz_target_h0_mm'] for i in np.where(m35)[0]])) if m35.any() else ''})
 for rank,i in enumerate(order[:20],1):
  if d[i]>0.75:break
  neigh.append({'state_id':s['state_id'],'rank':rank,'episode_id':int(E[i]),'frame_id':int(B[i]['frame_id']),
                'q_distance_rad':float(d[i]),'grip_mm':float(G[i]),'ee_z_mm':float(EZ[i]*1000),
                'phase':B[i]['phase_label'],'expert_dz_h0_mm':float(B[i]['dz_target_h0_mm']),
                'expert_dz_h9_mm':float(hmap.get((int(E[i]),int(B[i]['frame_id']),9),float('nan')))})
with (DATA/'queried_states.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(qrows[0].keys()));w.writeheader();w.writerows(qrows)
with (DATA/'local_joint_coverage.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(cov[0].keys()));w.writeheader();w.writerows(cov)
with (DATA/'local_common_support.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(cov[0].keys()));w.writeheader();w.writerows(cov)
with (DATA/'neighbor_expert_labels.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(neigh[0].keys()));w.writeheader();w.writerows(neigh)
print('states',len(states),'common0.3',sum(r['common_support'] for r in cov if r['radius_rad']==0.3),'common0.75',sum(r['common_support'] for r in cov if r['radius_rad']==0.75))
