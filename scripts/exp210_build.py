"""实验 2.10：D2 B90 逐帧联合表、阶段事件和全局夹爪统计。"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np, pybullet as p, pybullet_data, pyarrow.parquet as pq, pyarrow as pa

ROOT=Path('/scratch/lijh/simulate_vla'); SHARD=ROOT/'data/_shards_e23/B_D2'
OUT=ROOT/'results/experiment_2_10'; DATA=OUT/'data'; DATA.mkdir(parents=True,exist_ok=True)
H=[0,4,9,19,49]

def fk(qs):
 c=p.connect(p.DIRECT);p.setAdditionalSearchPath(pybullet_data.getDataPath(),physicsClientId=c)
 r=p.loadURDF('franka_panda/panda.urdf',useFixedBase=True,physicsClientId=c);out=[]
 for q in qs:
  for j in range(7):p.resetJointState(r,j,float(q[j]),physicsClientId=c)
  out.append(p.getLinkState(r,11,computeForwardKinematics=True,physicsClientId=c)[4])
 p.disconnect(c);return np.asarray(out)

def sustained(m,n=3):
 for i in range(max(0,len(m)-n+1)):
  if m[i:i+n].all():return i
 return None

def main():
 meta=[json.loads(x) for x in open(ROOT/'results/experiment_2_3/data/D2_episodes.jsonl',encoding='utf-8') if x.strip()]
 ok=[r for r in meta if r['success'] and r['reason']=='ok']
 rows=[];events=[];hor=[]
 for e in sorted((SHARD/'meta/episodes.jsonl').read_text().splitlines()):
  if not e.strip():continue
  ei=json.loads(e)['episode_index']
  if ei<60 or ei>=len(ok):continue
  d=pq.read_table(SHARD/f'data/chunk-000/episode_{ei:06d}.parquet',columns=['observation.state','action','timestamp']).to_pydict()
  st=np.asarray(d['observation.state'],float);ac=np.asarray(d['action'],float);ts=np.asarray(d['timestamp'],float);n=len(st)
  xyz=fk(st[:,:7]);ax=fk(ac[:,:7]);z=xyz[:,2];az=ax[:,2];dz=(az-z)*1000
  vz=np.zeros(n)
  for t in range(n-1):vz[t]=(z[t+1]-z[t])/(ts[t+1]-ts[t])*1000
  close=sustained(st[:,7:9].mean(1)<=0.016,2);des=sustained(dz<-5,3);low=sustained(z<0.05,2)
  events.append({'episode_id':ei,'n_frames':n,'initial_grip_mm':float(st[0,7:9].mean()*1000),
                 'close_cmd_t':float(ts[0]),'actual_close_t':float(ts[close]) if close is not None else '',
                 'descent_t':float(ts[des]) if des is not None else '', 'low_t':float(ts[low]) if low is not None else ''})
  for t in range(n):
   if close is not None and t<close:ph='initial'
   elif des is not None and t<des:ph='close_wait'
   elif low is not None and t<low:ph='descend'
   else:ph='push'
   rows.append({'episode_id':ei,'frame_id':t,'timestamp_s':float(ts[t]),'task':'B90','yaw_condition':'yaw90',
                'start_mode':ok[ei]['mode'],**{f'q{j+1}':float(st[t,j]) for j in range(7)},
                'finger_left':float(st[t,7]),'finger_right':float(st[t,8]),'gripper_per_finger_mm':float(st[t,7:9].mean()*1000),
                'ee_x':float(xyz[t,0]),'ee_y':float(xyz[t,1]),'ee_z':float(xyz[t,2]),
                'ee_orientation':'not_available',
                **{f'action_q{j+1}':float(ac[t,j]) for j in range(7)},'action_gripper':float(ac[t,7]),
                'expert_target_ee_x':float(ax[t,0]),'expert_target_ee_y':float(ax[t,1]),'expert_target_ee_z':float(ax[t,2]),
                'episode_progress':t/max(1,n-1),'phase_label':ph,'phase_label_source':'inferred',
                'time_since_gripper_close_command':float(ts[t]-ts[0]),
                'time_since_arm_descent_start':float(ts[t]-ts[des]) if des is not None else float('nan'),
                'object_pose_available':False,'object_relative_geometry_available':False,
                'dz_target_h0_mm':float(dz[t]),'vz_actual_mm_s':float(vz[t])})
   for h in H:
    if t+h<n:
     hor.append({'episode_id':ei,'frame_id':t,'timestamp_s':float(ts[t]),'horizon':h,'valid':1,
                 'gripper_per_finger_mm':float(st[t,7:9].mean()*1000),
                 'ee_z_mm':float(z[t]*1000),'expert_target_ee_z_mm':float(az[t+h]*1000),
                 'delta_z_mm':float((az[t+h]-z[t])*1000)})
  print('ep',ei,n,flush=True)
 pq.write_table(pa.Table.from_pylist(rows),DATA/'b90_frame_state_action.parquet')
 pq.write_table(pa.Table.from_pylist(hor),DATA/'expert_action_horizons.parquet')
 with (DATA/'expert_phase_events.csv').open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(events[0].keys()));w.writeheader();w.writerows(events)
 # global bins
 hmap={(x['episode_id'],x['frame_id'],x['horizon']):x['delta_z_mm'] for x in hor}
 G=np.array([r['gripper_per_finger_mm'] for r in rows]);Z=np.array([r['ee_z'] for r in rows])*1000
 bins=[(12.5,16.5,'narrow_12.5_16.5'),(20,30,'intermediate_20_30'),(34.5,35.5,'wide_34.5_35.5')]
 stats=[]
 for lo,hi,name in bins:
  m=(G>=lo)&(G<hi);sub=[r for r,k in zip(rows,m) if k]
  if not sub:continue
  row={'bin':name,'n_frames':len(sub),'n_episodes':len(set(r['episode_id'] for r in sub)),
       'grip_mean_mm':float(G[m].mean()),'ee_z_mean_mm':float(Z[m].mean()),'ee_z_min_mm':float(Z[m].min()),'ee_z_max_mm':float(Z[m].max()),
       'phase_initial':sum(r['phase_label']=='initial' for r in sub),'phase_close_wait':sum(r['phase_label']=='close_wait' for r in sub),
       'phase_descend':sum(r['phase_label']=='descend' for r in sub),'phase_push':sum(r['phase_label']=='push' for r in sub)}
  for h in H:
   vals=[]
   for i,r in enumerate(rows):
    if not m[i]:continue
    v=hmap.get((r['episode_id'],r['frame_id'],h))
    if v is not None:vals.append(v)
   row[f'dz_h{h}_mean_mm']=float(np.mean(vals)) if vals else ''
  stats.append(row)
 with (DATA/'global_gripper_action_stats.csv').open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(stats[0].keys()));w.writeheader();w.writerows(stats)
 print(json.dumps(stats,indent=1))

if __name__=='__main__':main()
