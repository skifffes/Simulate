import csv,json,sys,hashlib
from pathlib import Path
import numpy as np,pybullet as p
ROOT=Path('/scratch/lijh/simulate_vla');sys.path.insert(0,str(ROOT/'src'))
from vla.atomic_tasks.scene2 import build_scene
from vla.cameras import CameraRig
from vla.atomic_tasks.config2 import CAMERAS,IMAGE_SIZE
from manipulation import finger_widths,set_gripper
R=ROOT/'results/experiment_3_controlled';S=R/'initial_states';S.mkdir(parents=True,exist_ok=True);DATA=R/'data';DATA.mkdir(parents=True,exist_ok=True)
A=json.loads((ROOT/'results/experiment_3/data/a_terminal_states.csv').read_text()) if False else None
import csv as _csv
A=next(_csv.DictReader(open(ROOT/'results/experiment_3/data/a_terminal_states.csv',encoding='utf-8-sig')))
B=next(x for x in _csv.DictReader(open(ROOT/'results/experiment_3/data/b_training_start_states.csv',encoding='utf-8-sig')) if x['b_candidate_id']=='ep064_f000')
BREF=ROOT/'results/experiment_3/snapshots/b_reference/ep064_f000.json'
CJSON=ROOT/'results/experiment_3/snapshots/transition_endpoints/a90_40000000__ep064_f000.json'
qB=np.array([float(x) for x in B['q'].split(';')]);gw=float(B['grip_mm'])/1000
def pose(sim,block):
 ee=p.getLinkState(sim.robot,11,computeForwardKinematics=True,physicsClientId=sim.client)
 pos,quat=p.getBasePositionAndOrientation(block,physicsClientId=sim.client)
 vel,ang=p.getBaseVelocity(block,physicsClientId=sim.client)
 contacts=sim.collisions()
 return ee,pos,quat,vel,ang,contacts
def build_b():
 snap=json.loads((ROOT/A['bullet']).with_suffix('.json').read_text()) if (ROOT/A['bullet']).with_suffix('.json').exists() else None
 # scene seed from A
 sim,block,wall,tray,info=build_scene(seed=int(A['source_seed']),start=(0.55,0.18,0.02),jitter=0,yaw=0.0)
 p.restoreState(fileName=str(ROOT/A['bullet']),physicsClientId=sim.client)
 for j in range(7):p.resetJointState(sim.robot,j,float(qB[j]),physicsClientId=sim.client)
 for j in (9,10):p.resetJointState(sim.robot,j,gw,physicsClientId=sim.client)
 sim.command(qB);set_gripper(sim,gw,20,0.05)
 for _ in range(30):sim.step()
 bullet=S/'B_place.bullet';p.saveBullet(str(bullet),physicsClientId=sim.client)
 d={'seed':int(B['episode_id']),'bullet':str(bullet),'group':'yaw90','tag':'B_place','state_id':'B_place','kind':'B_place','source':'A scene qB reset diagnostic','obj':[float(x) for x in pose(sim,block)[1]],'width_m':gw,'q':qB.tolist()}
 (S/'B_place.json').write_text(json.dumps(d,indent=2));return sim,bullet
def capture(snap_path,out_tag,sim_override=None):
 snap=json.loads(Path(snap_path).read_text());sim,block,wall,tray,info=build_scene(seed=snap['seed'],start=(0.55,0.18,0.02),jitter=0,yaw=0.0)
 try:
  p.restoreState(fileName=str(Path(snap['bullet'])),physicsClientId=sim.client)
  qhold=sim.q();sim.command(qhold)
  for _ in range(30):sim.step()
  rig=CameraRig(sim,CAMERAS,IMAGE_SIZE,supersample=1);fr=rig.capture();ee,pos,quat,vel,ang,contacts=pose(sim,block)
  np.savez_compressed(S/f'{out_tag}_images.npz',top=fr['top'],wrist=fr['wrist'])
  q=sim.q();dq=np.array([p.getJointState(sim.robot,j,physicsClientId=sim.client)[1] for j in range(9)])
  row={'state':out_tag,'path':str(snap['bullet']),'q':';'.join(f'{v:.7f}' for v in q),'dq':';'.join(f'{v:.6f}' for v in dq),
       'ee_x':ee[4][0],'ee_y':ee[4][1],'ee_z':ee[4][2],'ee_quat':';'.join(f'{v:.6f}' for v in ee[5]),
       'obj_x':pos[0],'obj_y':pos[1],'obj_z':pos[2],'obj_quat':';'.join(f'{v:.6f}' for v in quat),
       'obj_vx':vel[0],'obj_vy':vel[1],'obj_vz':vel[2],'obj_wx':ang[0],'obj_wy':ang[1],'obj_wz':ang[2],
       'grip_left':float(finger_widths(sim)[0]),'grip_right':float(finger_widths(sim)[1]),'n_contacts':len(contacts),
       'top_sha':hashlib.sha1(fr['top'].tobytes()).hexdigest()[:12],'wrist_sha':hashlib.sha1(fr['wrist'].tobytes()).hexdigest()[:12]}
  return row
 finally:p.disconnect(sim.client)
if __name__=='__main__':
 sim,bullet=build_b();p.disconnect(sim.client)
 rows=[capture(BREF,'A_ref'),capture(S/'B_place.json','B_place'),capture(CJSON,'C_transition')]
 with (DATA/'initial_states.csv').open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
 print('ok')
