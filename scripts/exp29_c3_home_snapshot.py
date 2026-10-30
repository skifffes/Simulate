import json, sys
from pathlib import Path
import numpy as np, pybullet as p
ROOT=Path('/scratch/lijh/simulate_vla');sys.path.insert(0,str(ROOT/'src'))
from vla.atomic_tasks.scene2 import build_scene
from vla.atomic_tasks.starts import home_array, set_robot_state, ee_position
from manipulation import finger_widths, set_gripper
OUT=ROOT/'results/experiment_2_9/c3_home_width_10';OUT.mkdir(parents=True,exist_ok=True)
snap=json.loads((ROOT/'results/experiment_2_6/C3_direct_B90/states/G1/C3_snapshot/seed_40000000/c3_terminal.json').read_text())
sim,block,wall,tray,info=build_scene(seed=40000000,start=(0.55,0.18,0.02),jitter=0,yaw=np.pi/2)
try:
 p.restoreState(fileName=str(ROOT/snap['bullet']),physicsClientId=sim.client)
 b0,_=p.getBasePositionAndOrientation(block,physicsClientId=sim.client);b0=np.asarray(b0,float)
 set_robot_state(sim,home_array(),finger=0.035);sim.command(home_array());set_gripper(sim,0.035,20,0.05)
 for _ in range(60):sim.step()
 q=sim.q();ee=ee_position(sim);b1,_=p.getBasePositionAndOrientation(block,physicsClientId=sim.client);b1=np.asarray(b1,float)
 contacts=p.getContactPoints(bodyA=sim.robot,bodyB=block,physicsClientId=sim.client)
 move=float(np.linalg.norm(b1-b0)*1000)
 print('move_mm',move,'contacts',len(contacts),'ee',ee,'q',q)
 if move>0.5 or contacts:raise SystemExit('invalid')
 bullet=OUT/'c3_home.bullet';p.saveBullet(str(bullet),physicsClientId=sim.client)
 d={'seed':40000000,'bullet':str(bullet),'group':'yaw90','tag':'c3_home','state_id':'c3_home',
    'kind':'c3_home','source':'C3 object pose + HOME robot','obj':b1.tolist(),'obj_yaw':float(snap['obj_quat'][2])}
 (OUT/'c3_home.json').write_text(json.dumps(d,indent=2))
 print('saved',bullet)
finally:p.disconnect(sim.client)
