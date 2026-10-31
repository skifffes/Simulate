import csv,numpy as np
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
B=Path(r'D:\python\simulate\results\experiment_3');D=B/'data';F=B/'figures';F.mkdir(parents=True,exist_ok=True);FONT='C:/Windows/Fonts/arial.ttf'
def font(s):
 try:return ImageFont.truetype(FONT,s)
 except:return ImageFont.load_default()
# visual grid
npzs=sorted((B/'snapshots/transition_endpoints').glob('*_images.npz'))
W,H=1050,180+len(npzs)*270;img=Image.new('RGB',(W,H),'white');d=ImageDraw.Draw(img);d.text((20,10),'Transition endpoint vs B90 training reference (top / wrist)',font=font(15),fill='black')
for ri,f in enumerate(npzs):
 z=np.load(f);bid=f.stem.replace('a90_40000000__','').replace('_images','')
 y=40+ri*270;d.text((10,y),bid,font=font(10),fill='black')
 for ci,key in enumerate(['top','wrist']):
  ref=z[f'ref_{key}'];end=z[f'end_{key}'];sc=2
  ir=Image.fromarray(ref).resize((256,256));ie=Image.fromarray(end).resize((256,256))
  x=100+ci*430;img.paste(ir,(x,y+20));img.paste(ie,(x+270,y+20));d.text((x,y+2),f'{key}: reference',font=font(9),fill='black');d.text((x+270,y+2),f'{key}: endpoint',font=font(9),fill='black')
img.save(F/'fig05_visual_comparison.png')
# endpoint errors
E=list(csv.DictReader((D/'endpoint_matching.csv').open(encoding='utf-8-sig')))
img=Image.new('RGB',(1000,450),'white');d=ImageDraw.Draw(img);d.text((30,12),'Endpoint matching errors (all 3 pairs)',font=font(15),fill='black')
L,R,T,Bm=80,950,70,350;d.rectangle([L,T,R,Bm],outline='black')
for i,r in enumerate(E):
 x=80+(950-80)*(i+0.5)/len(E);q=float(r['joint_l2_error_rad']);ee=float(r['ee_position_error_mm'])
 py=350-(350-70)*min(1,q/0.05);d.rectangle([x-50,py,x-10,350],fill='#1f77b4')
 py2=350-(350-70)*min(1,ee/10);d.rectangle([x+10,py2,x+50,350],fill='#d62728')
 d.text((x-40,360),r['b_candidate_id'],font=font(8),fill='black')
img.save(F/'fig04_endpoint_errors.png')
# joint trajectory q1/q4 for one pair
tr=list(csv.DictReader((D/'execution_tracking.csv').open(encoding='utf-8-sig')))
sel=[r for r in tr if r['pair']=='a90_40000000__ep074_f000']
img=Image.new('RGB',(1000,450),'white');d=ImageDraw.Draw(img);d.text((30,12),'Executed joint trajectory: A90 -> ep074 f000',font=font(14),fill='black')
L,R,T,Bm=80,950,70,350;d.rectangle([L,T,R,Bm],outline='black')
for j,col in [(0,'#1f77b4'),(3,'#d62728')]:
 vals=[float(r['q'].split(';')[j]) for r in sel];lo,hi=min(vals),max(vals);pts=[]
 for i,v in enumerate(vals):
  x=80+(950-80)*i/max(1,len(vals)-1);y=350-(350-70)*(v-lo)/max(1e-6,hi-lo);pts.append((x,y))
 d.line(pts,fill=col,width=2)
img.save(F/'fig03_joint_trajectory.png')
# funnel
img=Image.new('RGB',(900,450),'white');d=ImageDraw.Draw(img);d.text((30,12),'Composability funnel',font=font(15),fill='black')
stages=['A unique','B starts','scene compat','goal valid','planned','executed','state match','visual match','B90 AtoB pairs'];vals=[1,60,60,3,3,3,3,3,2]
for i,(s,v) in enumerate(zip(stages,vals)):
 y=60+i*40;d.text((30,y),s,font=font(10),fill='black');d.rectangle([250,y,250+min(500,v*5),y+20],fill='#2ca02c');d.text((760,y),str(v),font=font(10),fill='black')
img.save(F/'fig07_composability_funnel.png')
print('done')
