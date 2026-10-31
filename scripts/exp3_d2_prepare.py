import csv,json,sys
from pathlib import Path
import numpy as np,pybullet as p
ROOT=Path('/scratch/lijh/simulate_vla');sys.path.insert(0,str(ROOT/'src'))
from vla.atomic_tasks.scene2 import build_scene
from manipulation import set_gripper
R=ROOT/'results/experiment_3';A=list(csv.DictReader((R/'data/a_terminal_states.csv').open(encoding='utf-8-sig')))[0]
B={r['b_candidate_id']:r for r in csv.DictReader((R/'data/b_training_start_states.csv').open(encoding='utf-8-sig'))}
for bid in ['ep074_f000','ep064_f000','ep070_f000']:
 b=B[bid];q=[float(x) for x in b['q'].split(';')];gw=float(b['grip_mm'])/1000
 # endpoint JSON
 ep=R/'snapshots/transition_endpoints'/f"{A['a_state_id']}__{bid}.json"
 ep.write_text(json.dumps({'seed':int(b['episode_id']),'bullet':str(R/'snapshots/transition_endpoints'/f"{A['a_state_id']}__{bid}.bullet"),
  'group':'yaw90','tag':f"AtoB_{bid}",'state_id':f"AtoB_{bid}",'kind':'transition','source':f"A90 {A['a_state_id']} -> {bid}",'obj':[float(b['obj_x']),float(b['obj_y']),float(b['obj_z'])],'width_m':gw},indent=2))
 # B reference bullet
 sim,block,wall,tray,info=build_scene(seed=int(b['scene_seed']),start=(float(b['obj_x']),float(b['obj_y']),float(b['obj_z'])),jitter=0,yaw=float(b['obj_yaw_rad']))
 try:
  for j in range(7):p.resetJointState(sim.robot,j,q[j],physicsClientId=sim.client)
  sim.command(q);set_gripper(sim,gw,20,0.05)
  for _ in range(20):sim.step()
  bullet=R/'snapshots/b_reference'/f'{bid}.bullet';bullet.parent.mkdir(parents=True,exist_ok=True);p.saveBullet(str(bullet),physicsClientId=sim.client)
  pos,_=p.getBasePositionAndOrientation(block,physicsClientId=sim.client)
  (R/'snapshots/b_reference'/f'{bid}.json').write_text(json.dumps({'seed':int(b['scene_seed']),'bullet':str(bullet),'group':'yaw90','tag':f'Bref_{bid}','state_id':f'Bref_{bid}','kind':'B_reference','source':bid,'obj':[float(x) for x in pos],'width_m':gw},indent=2))
 finally:p.disconnect(sim.client)
 print('ok',bid)
