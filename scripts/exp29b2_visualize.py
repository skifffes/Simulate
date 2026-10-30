import csv
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(r'D:\python\simulate');B=ROOT/'results/experiment_2_9/b2_low_near';F=B/'figures';F.mkdir(parents=True,exist_ok=True)
FONT='C:/Windows/Fonts/arial.ttf'
def font(s):
 try:return ImageFont.truetype(FONT,s)
 except:return ImageFont.load_default()
def axes(d,L,R,T,Bm,x0,x1,y0,y1,xl,yl):
 d.rectangle([L,T,R,Bm],outline='black')
 for i in range(5):
  y=Bm-(Bm-T)*i/4;d.line([L,y,R,y],fill=(225,225,225));d.text((8,y-6),f'{y0+(y1-y0)*i/4:.3g}',font=font(8),fill='black')
  x=L+(R-L)*i/4;d.text((x-12,Bm+6),f'{x0+(x1-x0)*i/4:.3g}',font=font(8),fill='black')
 d.text((L,Bm+20),xl,font=font(9),fill='black');d.text((L,T-12),yl,font=font(9),fill='black')
def pt(x,y,L,R,T,Bm,x0,x1,y0,y1):return (L+(R-L)*(x-x0)/(x1-x0),Bm-(Bm-T)*(y-y0)/(y1-y0))
orig=list(csv.DictReader((B/'data/original_low00_trajectories.csv').open(encoding='utf-8-sig')))
rec=list(csv.DictReader((B/'data/original_low00_recheck.csv').open(encoding='utf-8-sig')))
act=list(csv.DictReader((B/'data/actual_ee_trajectories.csv').open(encoding='utf-8-sig')))
out=list(csv.DictReader((B/'data/rollout_outcomes.csv').open(encoding='utf-8-sig')))
states=list(csv.DictReader((B/'data/state_manifest.csv').open(encoding='utf-8-sig')))
pred=list(csv.DictReader((B/'policy/policy_height_sensitivity.csv').open(encoding='utf-8-sig')))
# Fig1 original z(t)
img=Image.new('RGB',(1100,460),'white');d=ImageDraw.Draw(img);d.text((30,12),'Figure 1: original low_00 actual EE height',font=font(16),fill='black')
L,R,T,Bm=80,1000,70,360;axes(d,L,R,T,Bm,0,2.5,0,220,'time (s)','EE z (mm)')
for run,col in [('D2_14','#1f77b4'),('D2_35_r0','#d62728'),('E1_14','#2ca02c')]:
 sub=[r for r in orig if r['run']==run];d.line([pt(float(r['t']),float(r['ee_z'])*1000,L,R,T,Bm,0,2.5,0,220) for r in sub],fill=col,width=2)
for r in rec:
 if r['run'] in ('D2_14','D2_35_r0','E1_14') and r['contact_time_s']:
  x=pt(float(r['contact_time_s']),0,L,R,T,Bm,0,2.5,0,220)[0];d.line([x,T,x,Bm],fill='black',width=1)
img.save(F/'fig01_original_low00_trajectories.png')
# Fig2 grip timeline
img=Image.new('RGB',(1100,450),'white');d=ImageDraw.Draw(img);d.text((30,12),'Figure 2: original low_00 actual gripper opening',font=font(16),fill='black')
axes(d,80,1000,70,350,0,2.5,0,40,'time (s)','grip (mm)')
for run,col in [('D2_14','#1f77b4'),('D2_35_r0','#d62728')]:
 sub=[r for r in orig if r['run']==run];d.line([pt(float(r['t']),float(r['grip'])*1000,80,1000,70,350,0,2.5,0,40) for r in sub],fill=col,width=2)
img.save(F/'fig02_original_low00_gripper_timeline.png')
# Fig3 start distribution x-z
img=Image.new('RGB',(1000,470),'white');d=ImageDraw.Draw(img);d.text((30,12),'Figure 3: new start distribution (EE x-z)',font=font(16),fill='black')
axes(d,80,950,70,370,0.30,0.55,0,0.25,'EE x (m)','EE z (m)')
cols={'low_near':'#d62728','low_far':'#ff7f0e','mid_near':'#1f77b4'}
for r in states:
 x=float(r['ee_x']);z=float(r['ee_z']);c=cols[r['kind']];px,py=pt(x,z,80,950,70,370,0.30,0.55,0,0.25);d.ellipse([px-5,py-5,px+5,py+5],fill=c)
img.save(F/'fig03_new_start_distribution.png')
# Fig4 predicted vs actual low_near00 14/35
img=Image.new('RGB',(1100,470),'white');d=ImageDraw.Draw(img);d.text((30,12),'Figure 4: predicted (chunk) vs actual, low_near_00',font=font(16),fill='black')
axes(d,80,1000,70,370,0,2.5,0,220,'time (s)','EE z (mm)')
for w,col in [(14,'#1f77b4'),(35,'#d62728')]:
 sub=[r for r in act if r['state_id']=='low_near_00' and int(float(r['width_mm']))==w and r['condition']=='D2' and r['rep']=='0']
 d.line([pt(float(r['t']),float(r['ee_z'])*1000,80,1000,70,370,0,2.5,0,220) for r in sub],fill=col,width=2)
 # predicted mean from policy predictions; time = step*0.05
 p=[r for r in csv.DictReader((B/'policy/policy_predictions.csv').open(encoding='utf-8-sig')) if r['state_id']=='low_near_00' and int(r['width_mm'])==w]
 for step in [0,4,9,19,49]:
  vals=[float(r['ee_z']) for r in p if int(r['chunk_step'])==step];d.ellipse([pt(step*0.05,np.mean(vals)*1000,80,1000,70,370,0,2.5,0,220)[0]-4,pt(step*0.05,np.mean(vals)*1000,80,1000,70,370,0,2.5,0,220)[1]-4,pt(step*0.05,np.mean(vals)*1000,80,1000,70,370,0,2.5,0,220)[0]+4,pt(step*0.05,np.mean(vals)*1000,80,1000,70,370,0,2.5,0,220)[1]+4],outline=col)
img.save(F/'fig04_predicted_vs_actual.png')
# Fig5 z(x) original
img=Image.new('RGB',(1000,460),'white');d=ImageDraw.Draw(img);d.text((30,12),'Figure 5: original low_00 height vs x',font=font(16),fill='black')
axes(d,80,950,70,360,0.42,0.66,0,200,'EE x (m)','EE z (mm)')
for run,col in [('D2_14','#1f77b4'),('D2_35_r0','#d62728')]:
 sub=[r for r in orig if r['run']==run];d.line([pt(float(r['ee_x']),float(r['ee_z'])*1000,80,950,70,360,0.42,0.66,0,200) for r in sub],fill=col,width=2)
img.save(F/'fig05_height_vs_x.png')
# Fig6 outcomes by state/width D2
img=Image.new('RGB',(1100,470),'white');d=ImageDraw.Draw(img);d.text((30,12),'Figure 6: D2 success by state and width (B2)',font=font(15),fill='black')
axes(d,80,1000,70,360,0,15,0,1.1,'state x width','success rate')
states_order=['low_near_00','low_near_01','low_near_02','low_far_00','mid_near_00'];widths=[14,35,40]
for si,s in enumerate(states_order):
 for wi,w in enumerate(widths):
  vals=[r for r in out if r['condition']=='D2' and r['state_id']==s and int(float(r['width_mm']))==w]
  v=sum(r['success']=='True' for r in vals)/max(1,len(vals));px,py=pt(si*3+wi+0.5,v,80,1000,70,360,0,15,0,1.1);d.rectangle([px-22,py,px+22,360],fill='#d62728');d.text((px-10,360+8),str(w),font=font(8),fill='black')
 d.text((80+(1000-80)*(si*3+1.5)/15-25,390),s,font=font(8),fill='black')
img.save(F/'fig06_low_near_vs_controls.png')
print('done')
