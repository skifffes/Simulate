import csv,json,sys
from pathlib import Path
import numpy as np,pybullet as p,pybullet_data,pyarrow.parquet as pq
ROOT=Path('/scratch/lijh/simulate_vla');DATA=ROOT/'results/experiment_3/data';DATA.mkdir(parents=True,exist_ok=True)
ok=[json.loads(x) for x in open(ROOT/'results/experiment_2_3/data/D2_episodes.jsonl',encoding='utf-8') if x.strip()]
ok=[r for r in ok if r['success'] and r['reason']=='ok']
def fk(qs):
 c=p.connect(p.DIRECT);p.setAdditionalSearchPath(pybullet_data.getDataPath(),physicsClientId=c);r=p.loadURDF('franka_panda/panda.urdf',useFixedBase=True,physicsClientId=c);out=[]
 for q in qs:
  for j in range(7):p.resetJointState(r,j,float(q[j]),physicsClientId=c)
  s=p.getLinkState(r,11,computeForwardKinematics=True,physicsClientId=c);out.append((s[4],s[5]))
 p.disconnect(c);return out
rows=[]
for i in range(60,120):
 d=pq.read_table(ROOT/f'data/_shards_e23/B_D2/data/chunk-000/episode_{i:06d}.parquet',columns=['observation.state']).to_pydict()
 st=np.asarray(d['observation.state'],float);q=st[0,:7];ee,quat=fk([q])[0];r=ok[i]
 rows.append({'b_candidate_id':f'ep{i:03d}_f000','episode_id':i,'frame_id':0,'start_mode':r['mode'],'yaw':'yaw90',
              'q':';'.join(f'{v:.7f}' for v in q),'finger_left':float(st[0,7]),'finger_right':float(st[0,8]),
              'grip_mm':float(st[0,7:9].mean()*1000),'ee_x':float(ee[0]),'ee_y':float(ee[1]),'ee_z':float(ee[2]),
              'ee_quat':';'.join(f'{v:.6f}' for v in quat),
              'obj_x':float(r['scene']['block_start'][0]),'obj_y':float(r['scene']['block_start'][1]),'obj_z':float(r['scene']['block_start'][2]),
              'obj_yaw_rad':float(r['scene']['block_yaw']),'obj_quat':';'.join(f'{v:.6f}' for v in p.getQuaternionFromEuler([0,0,float(r['scene']['block_yaw'])])),
              'scene_seed':int(r['spec']['seed']),'episode_success':True})
with (DATA/'b_training_start_states.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
print(len(rows))
