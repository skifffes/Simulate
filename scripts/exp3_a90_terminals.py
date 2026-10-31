"""实验 3：生成真实 A90 成功终态快照。"""
import json,sys
from pathlib import Path
import numpy as np,pybullet as p
ROOT=Path('/scratch/lijh/simulate_vla');sys.path.insert(0,str(ROOT/'src'))
from vla.atomic_tasks.scene2 import build_scene,START,PLACE
from vla.atomic_tasks.experts import SkillExpert
OUT=ROOT/'results/experiment_3';S=OUT/'snapshots/a_terminals';S.mkdir(parents=True,exist_ok=True);DATA=OUT/'data';DATA.mkdir(parents=True,exist_ok=True)
rows=[]
for i,seed in enumerate([40000000,40000002,40000004,40000006,40000008]):
 sim,block,wall,tray,info=build_scene(seed=seed,start=START,jitter=0.0,yaw=np.pi/2)
 try:
  exp=SkillExpert(sim,block);res=exp.skill_a90()
  q=sim.q();pos,quat=p.getBasePositionAndOrientation(block,physicsClientId=sim.client);ee=p.getLinkState(sim.robot,11,computeForwardKinematics=True,physicsClientId=sim.client)[4]
  bullet=S/f'a90_{seed}.bullet';p.saveBullet(str(bullet),physicsClientId=sim.client)
  rows.append({'a_state_id':f'a90_{seed}','source_episode':None,'source_seed':seed,'task':'A90','success':bool(res.get('success')),
               'q':';'.join(f'{v:.7f}' for v in q),'finger_left':float(sim.q()[9] if False else p.getJointState(sim.robot,9,physicsClientId=sim.client)[0]),
               'finger_right':float(p.getJointState(sim.robot,10,physicsClientId=sim.client)[0]),
               'ee_x':float(ee[0]),'ee_y':float(ee[1]),'ee_z':float(ee[2]),
               'obj_x':float(pos[0]),'obj_y':float(pos[1]),'obj_z':float(pos[2]),
               'obj_quat':';'.join(f'{v:.7f}' for v in quat),'bullet':str(bullet)})
  print(seed,res.get('success'),res.get('reason'),pos,flush=True)
 finally:p.disconnect(sim.client)
import csv
with (DATA/'a_terminal_states.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
print('total',len(rows))
