"""实验 3 Pilot：真实 A90 终态 -> 真实 B90 episode 起点 的关节规划与执行。"""
import csv,json,sys,math,random
from pathlib import Path
import numpy as np,pybullet as p
ROOT=Path('/scratch/lijh/simulate_vla');sys.path.insert(0,str(ROOT/'src'))
from vla.atomic_tasks.scene2 import build_scene
from vla.cameras import CameraRig
from vla.atomic_tasks.config2 import CAMERAS,IMAGE_SIZE
from manipulation import finger_widths,set_gripper
OUT=ROOT/'results/experiment_3';DATA=OUT/'data';TRAJ=OUT/'trajectories/planned';SNAP=OUT/'snapshots/transition_endpoints'
for d in (DATA,TRAJ,SNAP,OUT/'trajectories/executed'):d.mkdir(parents=True,exist_ok=True)
A=list(csv.DictReader((DATA/'a_terminal_states.csv').open(encoding='utf-8-sig')))[0]
B={r['b_candidate_id']:r for r in csv.DictReader((DATA/'b_training_start_states.csv').open(encoding='utf-8-sig'))}
TARGETS=['ep074_f000','ep064_f000','ep070_f000'];STEP=0.18;SEED=7
random.seed(SEED)

def qarr(s):return np.array([float(x) for x in s.split(';')])
def ik_scene():
 s,b,w,t,info=build_scene(seed=int(A['source_seed']),start=(0.55,0.18,0.02),jitter=0,yaw=0.0);p.restoreState(fileName=str(ROOT/A['bullet']),physicsClientId=s.client);return s,b,w,t,info

def bounds(sim):return np.asarray(sim.lower),np.asarray(sim.upper)
def valid(sim,q):
 q=np.asarray(q);lo,hi=bounds(sim)
 if np.any(q<lo+1e-3) or np.any(q>hi-1e-3):return False
 for j in range(7):p.resetJointState(sim.robot,j,float(q[j]),physicsClientId=sim.client)
 p.performCollisionDetection(physicsClientId=sim.client)
 return not sim.collisions()
def edge(sim,a,b,n=16):
 for t in np.linspace(0,1,n):
  if not valid(sim,a*(1-t)+b*t):return False
 return True

def plan(sim,qa,qb,rng):
 if edge(sim,qa,qb,30):return [qa,qb],'straight'
 nodes=[qa];parent=[-1]
 for it in range(4000):
  if rng.random()<0.15:qrand=qb
  else:
   lo,hi=bounds(sim);qrand=rng.uniform(lo,hi)
  dist=[np.linalg.norm(np.asarray(n)-qrand) for n in nodes];ni=int(np.argmin(dist));near=np.asarray(nodes[ni])
  d=np.linalg.norm(qrand-near)
  if d<1e-6:continue
  step=min(STEP,d);new=near+(qrand-near)/d*step
  if not edge(sim,near,new):continue
  nodes.append(new);parent.append(ni)
  if np.linalg.norm(new-qb)<STEP*0.8 and edge(sim,new,qb,20):
   path=[qb];k=len(nodes)-1
   while k>=0:path.append(np.asarray(nodes[k]));k=parent[k]
   return list(reversed(path)),'rrt'
 return None,'fail'

def main():
 rng=random.Random(SEED);rows=[];execrows=[]
 for bid in TARGETS:
  b=B[bid];qa=qarr(A['q']);qb=qarr(b['q']);sim,block,wall,tray,info=ik_scene()
  try:
   # target validity in A scene
   goal_ok=valid(sim,qb)
   path,status=plan(sim,qa,qb,rng) if goal_ok else (None,'goal_collision')
   rows.append({'a_state_id':A['a_state_id'],'b_candidate_id':bid,'q_start':qa.tolist(),'q_goal':qb.tolist(),
                'goal_static_valid':bool(goal_ok),'planner':status,'plan_len':len(path) if path else 0,
                'scene_obj_diff_mm':float(np.linalg.norm(np.array([float(A['obj_x'])-float(b['obj_x']),float(A['obj_y'])-float(b['obj_y']),float(A['obj_z'])-float(b['obj_z'])])*1000)),
                'obj_yaw_diff_deg':''})
   if path:np.save(TRAJ/f'{A["a_state_id"]}__{bid}.npy',np.asarray(path))
   # execute: restore A fresh
   p.resetSimulation(physicsClientId=sim.client)
  finally:p.disconnect(sim.client)
  if not path:continue
  sim,block,wall,tray,info=ik_scene()
  try:
   q0=sim.q();obj0,_=p.getBasePositionAndOrientation(block,physicsClientId=sim.client);obj0=np.asarray(obj0)
   dense=[]
   for i in range(len(path)-1):
    a=np.asarray(path[i]);bseg=np.asarray(path[i+1]);n=max(2,int(np.ceil(np.linalg.norm(bseg-a)/0.01)))
    for t in np.linspace(0,1,n,endpoint=False):dense.append(a*(1-t)+bseg*t)
   dense.append(np.asarray(path[-1]))
   track=[]
   for q in dense:
    sim.command(q)
    for _ in range(12):sim.step()
    ee_t=p.getLinkState(sim.robot,11,computeForwardKinematics=True,physicsClientId=sim.client)[4]
    ob_t,_=p.getBasePositionAndOrientation(block,physicsClientId=sim.client)
    track.append({'step':len(track),'q':';'.join(f'{v:.7f}' for v in sim.q()),'ee_z':float(ee_t[2]),'obj_x':float(ob_t[0]),'obj_y':float(ob_t[1]),'obj_z':float(ob_t[2])})
   with (OUT/'trajectories/executed'/f'{A["a_state_id"]}__{bid}.csv').open('w',newline='',encoding='utf-8-sig') as tf:
    tw=csv.DictWriter(tf,fieldnames=list(track[0].keys()));tw.writeheader();tw.writerows(track)
   # set the recorded B90 starting gripper opening after arm path
   gw=float(b['grip_mm'])/1000.0
   for j in (9,10):p.resetJointState(sim.robot,j,gw,physicsClientId=sim.client)
   sim.command(qb);set_gripper(sim,gw,20.0,0.05)
   for _ in range(30):sim.step()
   qf=sim.q();objf,_=p.getBasePositionAndOrientation(block,physicsClientId=sim.client);objf=np.asarray(objf)
   ee_f=p.getLinkState(sim.robot,11,computeForwardKinematics=True,physicsClientId=sim.client)[4]
   # ref q ee
   sim2,blk2,_,_,_=build_scene(seed=int(b['scene_seed']),start=(float(b['obj_x']),float(b['obj_y']),float(b['obj_z'])),jitter=0,yaw=float(b['obj_yaw_rad']))
   try:
    for j in range(7):p.resetJointState(sim2.robot,j,float(qb[j]),physicsClientId=sim2.client)
    sim2.command(qb);ee_b=p.getLinkState(sim2.robot,11,computeForwardKinematics=True,physicsClientId=sim2.client)[4]
    rig_b=CameraRig(sim2,CAMERAS,IMAGE_SIZE,supersample=1);fr_b=rig_b.capture()
   finally:p.disconnect(sim2.client)
   rig=CameraRig(sim,CAMERAS,IMAGE_SIZE,supersample=1);fr=rig.capture()
   bullet=SNAP/f'{A["a_state_id"]}__{bid}.bullet';p.saveBullet(str(bullet),physicsClientId=sim.client)
   execrows.append({'a_state_id':A['a_state_id'],'b_candidate_id':bid,'joint_l2_error_rad':float(np.linalg.norm(qf-qb)),
                    'joint_max_abs_error_rad':float(np.max(np.abs(qf-qb))),'ee_position_error_mm':float(np.linalg.norm(np.asarray(ee_f)-np.asarray(ee_b))*1000),
                    'obj_drift_mm':float(np.linalg.norm(objf-obj0)*1000),'obj_z_drift_mm':float((objf[2]-obj0[2])*1000),
                    'top_mae':float(np.mean(np.abs(fr['top'].astype(float)-fr_b['top'].astype(float)))),
                    'wrist_mae':float(np.mean(np.abs(fr['wrist'].astype(float)-fr_b['wrist'].astype(float)))),
                    'snapshot':str(bullet),'ref_top':fr_b['top'],'ref_wrist':fr_b['wrist'],'end_top':fr['top'],'end_wrist':fr['wrist']})
   np.savez_compressed(SNAP/f'{A["a_state_id"]}__{bid}_images.npz',ref_top=fr_b['top'],ref_wrist=fr_b['wrist'],end_top=fr['top'],end_wrist=fr['wrist'])
   print('exec',bid,'qerr',execrows[-1]['joint_l2_error_rad'],'ee',execrows[-1]['ee_position_error_mm'],'obj',execrows[-1]['obj_drift_mm'],flush=True)
  finally:p.disconnect(sim.client)
 with (DATA/'planning_results.csv').open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
 with (DATA/'endpoint_matching.csv').open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=[k for k in execrows[0] if not k.endswith('top') and not k.endswith('wrist')]);w.writeheader();w.writerows([{k:v for k,v in r.items() if not k.endswith('top') and not k.endswith('wrist')} for r in execrows])
 with (DATA/'visual_matching.csv').open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=['a_state_id','b_candidate_id','top_mae','wrist_mae']);w.writeheader();w.writerows([{k:r[k] for k in ('a_state_id','b_candidate_id','top_mae','wrist_mae')} for r in execrows])
 print('done')
if __name__=='__main__':main()
