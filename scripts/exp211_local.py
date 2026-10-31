import csv,json
from pathlib import Path
import numpy as np,pyarrow as pa,pyarrow.parquet as pq
ROOT=Path(r'D:\python\simulate');B=ROOT/'results/experiment_2_11';D=B/'data'
err=list(csv.DictReader((D/'policy_expert_action_error.csv').open(encoding='utf-8-sig')))
meta=list(csv.DictReader((D/'observation_meta.csv').open(encoding='utf-8-sig')))
old={r['trajectory']:r for r in csv.DictReader((D/'gripper_condition_response.csv').open(encoding='utf-8-sig'))}
# teacher observations parquet
pq.write_table(pa.Table.from_pylist([{'trajectory':r['trajectory'],'frame':int(r['frame']),'seed':int(r['seed']),
 'q1':float(r['q1']),'q7':float(r['q7']),'grip_left':float(r['grip_left']),'grip_right':float(r['grip_right']),
 'expert_action_grip':float(r['expert_action_grip'])} for r in csv.DictReader((D/'teacher_forced_observations.csv').open(encoding='utf-8-sig'))]),D/'teacher_forced_observations.parquet')
# expert trajectories
pq.write_table(pa.Table.from_pylist([{**r,'frame':int(r['frame']),'time_s':float(r['time_s']),'grip_mm':float(r['grip_mm']),'ee_z_mm':float(r['ee_z_mm']),'expert_action_grip_mm':float(r['expert_action_grip_mm'])} for r in meta]),D/'expert_trajectories.parquet')
# response corrected
resp=[]
for tr,o in old.items():
 q0z=float(o['expert_z0_step0_mm'])-next(float(x['expert_dz_vs_obs_mm']) for x in err if x['trajectory']==tr and x['frame']=='0' and x['horizon']=='0')
 row={'trajectory':tr,'width_mm':int(o['width_mm']),'T':int(o['T']),'q0_ee_z_mm':q0z}
 for k in (0,4,9,19):
  ms=[float(x['model_dz_vs_obs_mm']) for x in err if x['trajectory']==tr and x['frame']=='0' and x['horizon']==str(k)]
  es=[float(x['expert_dz_vs_obs_mm']) for x in err if x['trajectory']==tr and x['frame']=='0' and x['horizon']==str(k)]
  row[f'model_z_step{k}_mm']=q0z+float(np.mean(ms)); row[f'expert_z_step{k}_mm']=q0z+float(np.mean(es))
  row[f'model_minus_expert_z_step{k}_mm']=float(np.mean(ms))-float(np.mean(es))
 resp.append(row)
with (D/'gripper_condition_response.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(resp[0].keys()));w.writeheader();w.writerows(resp)
# condition residual
res=[]
for state,pair in [('c3',('c3_g014','c3_g035')),('ep66',('ep66_g014','ep66_g035')),('c3_40',('c3_g014','c3_g040'))]:
 a=next(x for x in resp if x['trajectory']==pair[0]);b=next(x for x in resp if x['trajectory']==pair[1])
 for k in (0,4,9,19):
  md=float(b[f'model_z_step{k}_mm'])-float(a[f'model_z_step{k}_mm']);ed=float(b[f'expert_z_step{k}_mm'])-float(a[f'expert_z_step{k}_mm'])
  res.append({'state':state,'pair':f'{pair[0]} vs {pair[1]}','horizon':k,'model_condition_diff_mm':md,'expert_condition_diff_mm':ed,'residual_mm':md-ed})
with (D/'condition_response_residual.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(res[0].keys()));w.writeheader();w.writerows(res)
# phase summary
pmap={(r['trajectory'],int(r['frame'])):r['phase'] for r in meta}
ph=[]
for tr in old:
 for p in ('initial','descend','push'):
  for k in (0,4,9):
   sub=[x for x in err if x['trajectory']==tr and x['horizon']==str(k) and pmap.get((x['trajectory'],int(x['frame'])))==p]
   if sub:ph.append({'trajectory':tr,'phase':p,'horizon':k,'n':len(sub),'q_l2_mean':float(np.mean([float(x['q_l2']) for x in sub])),
                     'ee_z_err_mean_mm':float(np.mean([float(x['ee_z_err_mm']) for x in sub])),
                     'ee_xy_err_mean_mm':float(np.mean([float(x['ee_xy_err_mm']) for x in sub]))})
with (D/'phase_level_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(ph[0].keys()));w.writeheader();w.writerows(ph)
# C3 case
c3=[]
for tr in ('c3_g014','c3_g035','c3_g040'):
 r=next(x for x in resp if x['trajectory']==tr)
 c3.append({'state':'C3_seed40000000','width_mm':r['width_mm'],**{k:r[k] for k in r if k.startswith('model_z_step') or k.startswith('expert_z_step')}})
with (D/'c3_case_analysis.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(c3[0].keys()));w.writeheader();w.writerows(c3)
print('ok')
