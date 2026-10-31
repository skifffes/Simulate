import json,sys,csv
from pathlib import Path
import numpy as np,pybullet as p,cv2
from PIL import Image,ImageDraw,ImageFont
ROOT=Path('/scratch/lijh/simulate_vla');sys.path.insert(0,str(ROOT/'src'))
from vla.atomic_tasks.scene2 import build_scene
from vla.cameras import CameraRig
from vla.atomic_tasks.config2 import CAMERAS
from run_experiment_2_4 import make_view
R=ROOT/'results/experiment_3_controlled';V=R/'videos';V.mkdir(parents=True,exist_ok=True)
A=next(csv.DictReader(open(ROOT/'results/experiment_3/data/a_terminal_states.csv',encoding='utf-8-sig')))
path=np.load(ROOT/'results/experiment_3/trajectories/planned/a90_40000000__ep064_f000.npy')
sim,block,wall,tray,info=build_scene(seed=int(A['source_seed']),start=(0.55,0.18,0.02),jitter=0,yaw=0.0)
p.restoreState(fileName=str(ROOT/A['bullet']),physicsClientId=sim.client);q0=sim.q();sim.command(q0)
rig=CameraRig(sim,CAMERAS,(640,480),supersample=1);view,proj=make_view((0.55,-0.62,0.30),(0.60,0.16,0.10),fov=46.0)
dense=[]
for i in range(len(path)-1):
 a=np.asarray(path[i]);b=np.asarray(path[i+1]);n=max(2,int(np.ceil(np.linalg.norm(b-a)/0.01)))
 for t in np.linspace(0,1,n,endpoint=False):dense.append(a*(1-t)+b*t)
dense.append(np.asarray(path[-1]))
vw=cv2.VideoWriter(str(V/'AtoB_transition.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(1280,520))
def frame(i,total):
 fr=rig.capture();w,h=640,480
 vm=p.computeViewMatrixFromYawPitchRoll if False else None
 vmat=p.computeViewMatrix([0.55,-0.62,0.30],[0.60,0.16,0.10],[0,0,1])
 pm=p.computeProjectionMatrixFOV(46,640/480,0.01,2.0)
 _,_,rgb,_,_=p.getCameraImage(640,480,viewMatrix=vmat,projectionMatrix=pm,renderer=p.ER_BULLET_HARDWARE_OPENGL,physicsClientId=sim.client)
 side=Image.fromarray(np.reshape(rgb,(480,640,4))[:,:,:3][:,:,::-1].astype(np.uint8) if np.reshape(rgb,(480,640,4)).shape[:2]==(480,640) else np.reshape(rgb,(640,480,4))[:,:,:3][:,:,::-1].astype(np.uint8)).resize((640,480))
 top=Image.fromarray(fr['top']).resize((640,480))
 canvas=np.ones((520,1280,3),np.uint8)*255
 canvas[20:500,0:640]=np.asarray(side);canvas[20:500,640:1280]=np.asarray(top)
 img=Image.fromarray(canvas);d=ImageDraw.Draw(img);d.text((10,4),'A90 terminal -> dataset B90 start (ep064 f000)',fill='black');d.text((650,4),f'transition {i+1}/{total}',fill='black')
 return np.asarray(img)
for i,q in enumerate(dense):
 sim.command(q)
 for _ in range(12):sim.step()
 if i%3==0:
  img=frame(i,len(dense));vw.write(img[:,:,::-1])
vw.release();print('saved',V/'AtoB_transition.mp4')
