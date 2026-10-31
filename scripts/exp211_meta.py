import csv,glob,numpy as np,pybullet as p,pybullet_data
from pathlib import Path
ROOT=Path('/scratch/lijh/simulate_vla');OUT=ROOT/'results/experiment_2_11/data';OUT.mkdir(parents=True,exist_ok=True)
def fk(qs):
 c=p.connect(p.DIRECT);p.setAdditionalSearchPath(pybullet_data.getDataPath(),physicsClientId=c);r=p.loadURDF('franka_panda/panda.urdf',useFixedBase=True,physicsClientId=c);out=[]
 for q in qs:
  for j in range(7):p.resetJointState(r,j,float(q[j]),physicsClientId=c)
  out.append(p.getLinkState(r,11,computeForwardKinematics=True,physicsClientId=c)[4])
 p.disconnect(c);return np.asarray(out)
rows=[]
for f in sorted((ROOT/'results/experiment_2_11/observations').glob('*.npz')):
 d=np.load(f);z=fk(d['q'])[:,2]
 for t in range(len(d['q'])):
  g=float(d['grip'][t].mean()*1000);ph='initial' if g>16 else ('descend' if z[t]>0.05 else 'push')
  rows.append({'trajectory':f.stem,'frame':t,'time_s':float(d['time'][t]),'grip_mm':g,'ee_z_mm':float(z[t]*1000),
               'phase':ph,'phase_source':'inferred','expert_action_grip_mm':float(d['action'][t,7]*1000)})
with (OUT/'observation_meta.csv').open('w',newline='',encoding='utf-8-sig') as fh:
 w=csv.DictWriter(fh,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
print(len(rows))
