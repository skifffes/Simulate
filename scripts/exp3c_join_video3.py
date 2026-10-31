import cv2,numpy as np
from pathlib import Path
V=Path('/scratch/lijh/simulate_vla/results/experiment_3_controlled/videos');C=Path('/scratch/lijh/simulate_vla/results/experiment_3_controlled/rollout_v2/C_continuous_r0')
def read(p):
 cap=cv2.VideoCapture(str(p));out=[]
 while True:
  ok,fr=cap.read()
  if not ok:break
  out.append(fr)
 cap.release();return out
a=read(V/'AtoB_transition.mp4');side=read(C/'side.mp4');top=read(C/'top.mp4')
b=[]
for i,(s,t) in enumerate(zip(side,top)):
 canvas=np.ones((520,1280,3),np.uint8)*255
 canvas[20:500,0:640]=cv2.resize(s,(640,480));canvas[20:500,640:1280]=cv2.resize(t,(640,480))
 cv2.putText(canvas,'B90 immediately after continuous transition (seed 0)',(10,14),cv2.FONT_HERSHEY_SIMPLEX,0.42,(0,0,0),1,cv2.LINE_AA)
 cv2.putText(canvas,f'tick {i}',(1150,14),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,0),1,cv2.LINE_AA)
 b.append(canvas)
w=cv2.VideoWriter(str(V/'AtoB_full.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(1280,520))
for fr in a+b:w.write(fr)
w.release();print(len(a),len(b))
for i,name in [(0,'start'),(40,'transition'),(60,'b90'),(len(a)+len(b)-1,'end')]:
 cap=cv2.VideoCapture(str(V/'AtoB_full.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,fr=cap.read();cv2.imwrite(str(V/f'frame_{name}.png'),fr);cap.release()
