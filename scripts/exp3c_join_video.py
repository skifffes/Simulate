import cv2,os
from pathlib import Path
V=Path('/scratch/lijh/simulate_vla/results/experiment_3_controlled/videos')
C=Path('/scratch/lijh/simulate_vla/results/experiment_3_controlled/rollout/C_r0/frozen/AtoB_ep064_f000/comparison.mp4')
def frames(p):
 cap=cv2.VideoCapture(str(p));out=[]
 while True:
  ok,fr=cap.read()
  if not ok:break
  out.append(cv2.resize(fr,(1280,520)))
 cap.release();return out
a=frames(V/'AtoB_transition.mp4');b=frames(C)
print(len(a),len(b))
w=cv2.VideoWriter(str(V/'AtoB_full.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(1280,520))
for fr in a+b:w.write(fr)
w.release();print('saved',V/'AtoB_full.mp4')
