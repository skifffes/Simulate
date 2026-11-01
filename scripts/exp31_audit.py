import csv,json,math,sys
from pathlib import Path
import numpy as np,pybullet as p
ROOT=Path('/scratch/lijh/simulate_vla');sys.path.insert(0,str(ROOT/'src'))
from vla.atomic_tasks.scene2 import build_scene
from manipulation import finger_widths,set_gripper
R=ROOT/'results/experiment_3_2';R.mkdir(parents=True,exist_ok=True)
A=next(csv.DictReader(open(ROOT/'results/experiment_3/data/a_terminal_states.csv',encoding='utf-8-sig')))
B=list(csv.DictReader(open(ROOT/'results/experiment_3/data/b_training_start_states.csv',encoding='utf-8-sig')))
qa=np.array([float(x) for x in A['q'].split(';')]);oa=np.array([float(x) for x in A['obj_quat'].split(';')])
def qmul(a,b):
 ax,ay,az,aw=a;bx,by,bz,bw=b
 return np.array([aw*bx+ax*bw+ay*bz-az*by,aw*by-ax*bz+ay*bw+az*bx,aw*bz+ax*by-ay*bx+az*bw,aw*bw-ax*bx-ay*by-az*bz])
def qconj(q):return np.array([-q[0],-q[1],-q[2],q[3]])
def qrot(q,v):return qmul(qmul(q,np.array([*v,0])),qconj(q))[:3]
def qang(a,b):return math.degrees(2*math.acos(min(1,abs(float(np.dot(a,b))))))
def yaw(q):
 x,y,z,w=q;return math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
def rel(ee,eeq,obj,objq):return qrot(qconj(eeq),np.array(obj)-np.array(ee)),qmul(qconj(eeq),np.array(objq))
# FK A EE
c=p.connect(p.DIRECT);p.setAdditionalSearchPath(__import__('pybullet_data').getDataPath(),physicsClientId=c);robot=p.loadURDF('franka_panda/panda.urdf',useFixedBase=True,physicsClientId=c)
for j in range(7):p.resetJointState(robot,j,float(qa[j]),physicsClientId=c)
eeA=p.getLinkState(robot,11,computeForwardKinematics=True,physicsClientId=c);eeA=np.array(eeA[4]);eqA=np.array(eeA[5]) if False else np.array(p.getLinkState(robot,11,computeForwardKinematics=True,physicsClientId=c)[5]);p.disconnect(c)
# A scene collision checker
sim,block,wall,tray,info=build_scene(seed=int(A['source_seed']),start=(0.55,0.18,0.02),jitter=0,yaw=0.0);p.restoreState(fileName=str(ROOT/A['bullet']),physicsClientId=sim.client)
def valid(q):
 for j in range(7):p.resetJointState(sim.robot,j,float(q[j]),physicsClientId=sim.client)
 p.performCollisionDetection(physicsClientId=sim.client);return not sim.collisions()
def edge(a,b,n=30):
 for t in np.linspace(0,1,n):
  if not valid(a*(1-t)+b*t):return False
 return True
rows=[]
for r in B:
 qb=np.array([float(x) for x in r['q'].split(';')]);ob=np.array([float(x) for x in r['obj_quat'].split(';')]);pb=np.array([float(r['ee_x']),float(r['ee_y']),float(r['ee_z'])]);
 pa,qa_rel=rel(eeA,eqA,[float(A['obj_x']),float(A['obj_y']),float(A['obj_z'])],oa)
 pb2,qb_rel=rel(pb,[float(x) for x in r['ee_quat'].split(';')],[float(r['obj_x']),float(r['obj_y']),float(r['obj_z'])],ob)
 axis=abs(math.degrees(math.atan2(math.sin(yaw(oa)-yaw(ob)),math.cos(yaw(oa)-yaw(ob)))));axis=min(axis,180-axis)
 rows.append({'b_candidate_id':r['b_candidate_id'],'episode_id':r['episode_id'],'start_mode':r['start_mode'],'q_dist_rad':float(np.linalg.norm(qb-qa)),
  'obj_pos_diff_mm':float(np.linalg.norm([float(A['obj_x'])-float(r['obj_x']),float(A['obj_y'])-float(r['obj_y']),float(A['obj_z'])-float(r['obj_z'])])*1000),
  'obj_rot_deg':qang(oa,ob),'obj_axis_deg':axis,'grip_mm':float(r['grip_mm']),'grip_diff_mm':abs(float(r['grip_mm'])-float(A['finger_left'])*1000),
  'goal_valid_A':bool(valid(qb)),'path_valid_A':bool(edge(qa,qb)),'rel_pos_diff_mm':float(np.linalg.norm(pa-pb2)*1000),'rel_rot_diff_deg':qang(qa_rel,qb_rel),
  'a_obj_x':A['obj_x'],'a_obj_y':A['obj_y'],'b_obj_x':r['obj_x'],'b_obj_y':r['obj_y']})
p.disconnect(sim.client)
with open(R/'b90_all_candidates.csv','w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
print(len(rows),sum(x['goal_valid_A'] for x in rows),sum(x['path_valid_A'] for x in rows))
