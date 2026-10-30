import csv,sys
from pathlib import Path
import numpy as np,pybullet as p
ROOT=Path('/scratch/lijh/simulate_vla');sys.path.insert(0,str(ROOT/'src'))
from vla.atomic_tasks.scene2 import build_scene
from manipulation import finger_widths,set_gripper
rows=list(csv.DictReader((ROOT/'results/experiment_2_9/b2_low_near/data/state_manifest.csv').open(encoding='utf-8-sig')))
out=[]
for s in rows:
 for w in (0.014,0.035,0.040):
  sim,block,wall,tray,info=build_scene(seed=int(s['seed']),start=(0.5,0,0),jitter=0,yaw=0.0)
  try:
   p.restoreState(fileName=str(ROOT/s['bullet']),physicsClientId=sim.client);q=sim.q();sim.command(q)
   b0,_=p.getBasePositionAndOrientation(block,physicsClientId=sim.client)
   for j in (9,10):p.resetJointState(sim.robot,j,w,physicsClientId=sim.client)
   sim.command(q);set_gripper(sim,w,20,0.05)
   for _ in range(10):sim.step()
   b1,_=p.getBasePositionAndOrientation(block,physicsClientId=sim.client)
   c=p.getContactPoints(bodyA=sim.robot,bodyB=block,physicsClientId=sim.client)
   out.append({'state_id':s['state_id'],'kind':s['kind'],'width_m':w,'actual_m':float(np.mean(finger_widths(sim))),
               'obj_move_mm':float(np.linalg.norm(np.asarray(b1)-np.asarray(b0))*1000),'block_contacts':len(c),
               'valid':bool(not c and float(np.linalg.norm(np.asarray(b1)-np.asarray(b0))*1000)<0.5)})
  finally:p.disconnect(sim.client)
with (ROOT/'results/experiment_2_9/b2_low_near/data/state_physical_validity.csv').open('w',newline='',encoding='utf-8-sig') as f:
 wri=csv.DictWriter(f,fieldnames=list(out[0].keys()));wri.writeheader();wri.writerows(out)
print(len(out),sum(r['valid'] for r in out))
