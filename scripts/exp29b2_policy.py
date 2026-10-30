"""实验 2.9-B2 A3：新 Low-Near/Low-Far/Mid-Near 的 14/35/40 mm 预测。"""
from __future__ import annotations
import argparse,csv,sys
from pathlib import Path
import numpy as np, pybullet as p, pybullet_data, torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from vla.atomic_tasks.scene2 import build_scene
from vla.atomic_tasks.config2 import CAMERAS,IMAGE_SIZE,TASKS
from vla.cameras import CameraRig
from manipulation import set_gripper
WIDTHS=[0.014,0.035,0.040];H=[0,4,9,19,49]
class FK:
 def __init__(self):
  self.c=p.connect(p.DIRECT);p.setAdditionalSearchPath(pybullet_data.getDataPath(),physicsClientId=self.c);self.r=p.loadURDF('franka_panda/panda.urdf',useFixedBase=True,physicsClientId=self.c)
 def xyz(self,qs):
  out=[]
  for q in qs:
   for j in range(7):p.resetJointState(self.r,j,float(q[j]),physicsClientId=self.c)
   out.append(p.getLinkState(self.r,11,computeForwardKinematics=True,physicsClientId=self.c)[4])
  return np.asarray(out)
 def close(self):p.disconnect(self.c)
def batch(policy,frames,state):
 b={}
 for n in CAMERAS:
  x=torch.from_numpy(np.ascontiguousarray(frames[n])).to(policy.device).float().div_(255.0);b[f'observation.images.{n}']=x.permute(2,0,1).unsqueeze(0)
 b['observation.state']=torch.from_numpy(np.asarray(state,np.float32)).to(policy.device).unsqueeze(0);b['task']=[TASKS['B']];return b
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--policy',required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--seeds',type=int,default=10);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 states=list(csv.DictReader(a.manifest.open(encoding='utf-8-sig')));policy=SmolVLAPolicy.from_pretrained(a.policy);policy.to('cuda');policy.eval();policy.device=next(policy.parameters()).device;fk=FK();pred=[];sens=[]
 for s in states:
  sim,block,wall,tray,info=build_scene(seed=int(s['seed']),start=(0.5,0,0),jitter=0,yaw=0.0)
  try:
   p.restoreState(fileName=str(ROOT/s['bullet']),physicsClientId=sim.client);q=np.asarray(sim.q(),float);frames={}
   for w in WIDTHS:
    for j in (9,10):p.resetJointState(sim.robot,j,w,physicsClientId=sim.client)
    sim.command(q);set_gripper(sim,w,20,0.05)
    for _ in range(10):sim.step()
    frames[w]=CameraRig(sim,CAMERAS,IMAGE_SIZE,supersample=1).capture()
  finally:p.disconnect(sim.client)
  chunks=[];zs=[]
  for w in WIDTHS:
   cw=[];zw=[]
   for seed in range(a.seeds):
    policy.reset();torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    with torch.no_grad():ch=policy.predict_action_chunk(batch(policy,frames[w],np.concatenate([q,[w,w]]))).squeeze(0).float().cpu().numpy()
    cw.append(ch);zw.append(fk.xyz(ch[:,:7]))
   chunks.append(np.asarray(cw));zs.append(np.asarray(zw))
  chunks=np.asarray(chunks);zs=np.asarray(zs);np.savez_compressed(a.out/f"{s['state_id']}.npz",chunks=chunks,xyz=zs,q=q,seed_count=a.seeds)
  for wi,w in enumerate(WIDTHS):
   for seed in range(a.seeds):
    for k in range(50):
     pred.append({'state_id':s['state_id'],'kind':s['kind'],'width_mm':round(w*1000),'seed':seed,'chunk_step':k,
                  'ee_x':float(zs[wi,seed,k,0]),'ee_y':float(zs[wi,seed,k,1]),'ee_z':float(zs[wi,seed,k,2]),'gripper_target':float(chunks[wi,seed,k,7])})
  for k in H:
   for lo,hi,tag in ((0.014,0.035,'14-35'),(0.035,0.040,'35-40'),(0.014,0.040,'14-40')):
    li=WIDTHS.index(lo);hi_i=WIDTHS.index(hi);dz=(zs[hi_i,:,k,2]-zs[li,:,k,2])*1000
    sens.append({'state_id':s['state_id'],'kind':s['kind'],'intervention':tag,'chunk_step':k,'mean_delta_z_mm':float(dz.mean()),'std_delta_z_mm':float(dz.std()),'positive_fraction':float((dz>0).mean()),'num_seeds':a.seeds})
  print('done',s['state_id'],flush=True)
 with (a.out/'policy_predictions.csv').open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(pred[0].keys()));w.writeheader();w.writerows(pred)
 with (a.out/'policy_height_sensitivity.csv').open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(sens[0].keys()));w.writeheader();w.writerows(sens)
 fk.close()
if __name__=='__main__':main()
