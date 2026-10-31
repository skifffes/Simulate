"""实验 2.11：在专家观测上 teacher-force 冻结 D2，并与专家动作标签比较。"""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np,pybullet as p,pybullet_data,torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from vla.atomic_tasks.config2 import CAMERAS,TASKS
def fk(qs):
 c=p.connect(p.DIRECT);p.setAdditionalSearchPath(pybullet_data.getDataPath(),physicsClientId=c);r=p.loadURDF('franka_panda/panda.urdf',useFixedBase=True,physicsClientId=c);pos=[];quat=[]
 for q in qs:
  for j in range(7):p.resetJointState(r,j,float(q[j]),physicsClientId=c)
  s=p.getLinkState(r,11,computeForwardKinematics=True,physicsClientId=c);pos.append(s[4]);quat.append(s[5])
 p.disconnect(c);return np.asarray(pos),np.asarray(quat)
def batch(policy,top,wrist,state):
 b={}
 for n,img in (('top',top),('wrist',wrist)):
  x=torch.from_numpy(np.ascontiguousarray(img)).to(policy.device).float().div_(255.0);b[f'observation.images.{n}']=x.permute(2,0,1).unsqueeze(0)
 b['observation.state']=torch.from_numpy(np.asarray(state,np.float32)).to(policy.device).unsqueeze(0);b['task']=[TASKS['B']];return b
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--obs',type=Path,required=True);ap.add_argument('--policy',required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--seeds',type=int,default=3);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 policy=SmolVLAPolicy.from_pretrained(a.policy);policy.to('cuda');policy.eval();policy.device=next(policy.parameters()).device
 errs=[];obsrows=[];acts={};resp=[]
 for f in sorted(a.obs.glob('*.npz')):
  d=np.load(f);top=d['top'];wrist=d['wrist'];q=d['q'];grip=d['grip'];ea=d['action'];T=len(q);name=f.stem
  ep=t if False else None
  zobs,_=fk(q);eap,_=fk(ea[:,:7]);ez=eap[:,2]
  chunks=[]
  for s in range(a.seeds):
   pol=[]
   for t in range(T):
    policy.reset();torch.manual_seed(s);torch.cuda.manual_seed_all(s)
    st=np.concatenate([q[t],grip[t]])
    with torch.no_grad():ch=policy.predict_action_chunk(batch(policy,top[t],wrist[t],st)).squeeze(0).float().cpu().numpy()
    pol.append(ch)
    if t in (0,1,2):
     obsrows.append({'trajectory':name,'frame':t,'seed':s,'q1':q[t,0],'q7':q[t,6],'grip_left':grip[t,0],'grip_right':grip[t,1],'expert_action_grip':ea[t,7]})
   pol=np.asarray(pol);chunks.append(pol)
   K=pol.shape[1]
   pz,pq=fk(pol[:,:,:7].reshape(-1,7));pz=pz.reshape(T,K,3);pq=pq.reshape(T,K,4)
   for t in range(0,T,1):
    for k in (0,4,9,19,49):
     if k>=K or t+k>=T:continue
     qerr=float(np.linalg.norm(pol[t,k,:7]-ea[t+k,:7]))
     ezerr=float((pz[t,k,2]-eap[t+k,2])*1000);exy=float(np.linalg.norm(pz[t,k,:2]-eap[t+k,:2])*1000)
     dot=abs(float(np.dot(pq[t,k],eap[0,0]*0+np.array([0,0,0,1])))) if False else 0.0
     errs.append({'trajectory':name,'frame':t,'seed':s,'horizon':k,'q_l2':qerr,'ee_z_err_mm':ezerr,'ee_xy_err_mm':exy,
                  'grip_err_mm':float((pol[t,k,7]-ea[t+k,7])*1000),
                  'model_dz_vs_obs_mm':float((pz[t,k,2]-zobs[t,2])*1000),
                  'expert_dz_vs_obs_mm':float((eap[t+k,2]-zobs[t,2])*1000)})
  chunks=np.asarray(chunks);acts[name]=chunks
  # initial condition response at t=0 relative to expert initial state
  resp.append({'trajectory':name,'width_mm':int(round(float(d['width'])*1000)),'T':T,
               'model_z0_step0_mm':float(chunks[:,0,0,2].mean()*1000),
               'model_z0_step4_mm':float(chunks[:,0,min(4,K-1),2].mean()*1000),
               'model_z0_step9_mm':float(chunks[:,0,min(9,K-1),2].mean()*1000),
               'model_z0_step19_mm':float(chunks[:,0,min(19,K-1),2].mean()*1000),
               'expert_z0_step0_mm':float(eap[0,2]*1000),'expert_z0_step4_mm':float(eap[4,2]*1000),
               'expert_z0_step9_mm':float(eap[9,2]*1000),'expert_z0_step19_mm':float(eap[19,2]*1000),
               'expert_grip_action0_mm':float(ea[0,7]*1000)})
  print('done',name,T,flush=True)
 np.savez_compressed(a.out/'teacher_forced_policy_actions.npz',**acts)
 with (a.out/'policy_expert_action_error.csv').open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(errs[0].keys()));w.writeheader();w.writerows(errs)
 with (a.out/'gripper_condition_response.csv').open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(resp[0].keys()));w.writeheader();w.writerows(resp)
 with (a.out/'teacher_forced_observations.csv').open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(obsrows[0].keys()));w.writeheader();w.writerows(obsrows)
if __name__=='__main__':main()
