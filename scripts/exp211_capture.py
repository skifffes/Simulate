"""实验 2.11：在专家轨迹上捕获 teacher-forcing 所需观测。"""
import argparse,json,sys
from pathlib import Path
import numpy as np,pybullet as p
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from vla.atomic_tasks.scene2 import build_scene
from vla.atomic_tasks.config2 import CAMERAS,IMAGE_SIZE
from vla.cameras import CameraRig
from manipulation import finger_widths,set_gripper
from run_experiment_2_4 import SPA
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--snap',type=Path,required=True);ap.add_argument('--width',type=float,required=True);ap.add_argument('--trace',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 snap=json.loads(a.snap.read_text());rows=[json.loads(x) for x in a.trace.read_text().splitlines() if x.strip()];acts=[r for r in rows if r.get('skill')=='B']
 sim,block,wall,tray,info=build_scene(seed=snap['seed'],start=(0.55,0.18,0.02),jitter=0,yaw=0.0)
 try:
  p.restoreState(fileName=str(ROOT/snap['bullet']),physicsClientId=sim.client)
  q=sim.q();sim.command(q)
  for j in (9,10):p.resetJointState(sim.robot,j,a.width,physicsClientId=sim.client)
  set_gripper(sim,a.width,20,0.05)
  for _ in range(10):sim.step()
  rig=CameraRig(sim,CAMERAS,IMAGE_SIZE,supersample=1)
  tops=[];wrists=[];qs=[];grips=[];targets=[]
  for r in acts:
   fr=rig.capture();tops.append(fr['top']);wrists.append(fr['wrist']);qs.append(np.asarray(sim.q(),float));grips.append(np.asarray(finger_widths(sim),float));targets.append(np.asarray(r['action'],float))
   act=targets[-1];sim.command(act[:7]);set_gripper(sim,float(act[7]),20,0.05)
   for _ in range(SPA):sim.step()
  a.out.parent.mkdir(parents=True,exist_ok=True)
  np.savez_compressed(a.out,top=np.asarray(tops),wrist=np.asarray(wrists),q=np.asarray(qs),grip=np.asarray(grips),action=np.asarray(targets),time=np.asarray([r['physics_time_sec'] for r in acts]),width=a.width)
  print('saved',a.out,len(acts))
 finally:p.disconnect(sim.client)
if __name__=='__main__':main()
