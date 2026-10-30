import csv
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
ROOT=Path(r'D:\python\simulate'); B=ROOT/'results/experiment_2_9/ab'; F=B/'figures'; F.mkdir(parents=True,exist_ok=True)
rows=list(csv.DictReader((B/'data/all_rollouts.csv').open(encoding='utf-8-sig')))
comp=list(csv.DictReader((B/'data/expert_policy_comparison.csv').open(encoding='utf-8-sig')))
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
states=['low_00','low_03','high_00','high_03']; widths=[0.014,0.035,0.040]
# B1 success rates
img=Image.new('RGB',(1100,480),'white');d=ImageDraw.Draw(img);d.text((30,12),'B1: expert vs frozen D2 success by state and initial width',font=font(15),fill='black')
L,R,T,Bm=80,1000,70,360;axes(d,L,R,T,Bm,0,len(states)*3,0,1.05,'state x width','success rate')
for si,s in enumerate(states):
 for wi,w in enumerate(widths):
  x=si*3+wi+0.5
  v=float(next(c for c in comp if c['state_id']==s and abs(float(c['width_m'])-w)<1e-6)['D2_success_rate'])
  px,py=pt(x,v,L,R,T,Bm,0,12,0,1.05);d.rectangle([px-25,py,px+25,Bm],fill='#d62728')
  d.text((px-10,Bm+8),f'{w*1000:.0f}',font=font(8),fill='black')
 d.text((L+(R-L)*(si*3+1.5)/12-20,Bm+30),s,font=font(9),fill='black')
d.text((100,430),'D2: low00 14=2/2,35=0/2,40=1/2; low03 14=2/2,35=1/2,40=2/2; high00 14=1/2,35=2/2,40=2/2; high03 14=1/2,35=1/2,40=2/2',font=font(8),fill='black')
img.save(F/'figB1_success.png')
# B2 lift vs tilt
img=Image.new('RGB',(1000,470),'white');d=ImageDraw.Draw(img);d.text((30,12),'B2: D2 failures vs expert (max lift / final tilt)',font=font(15),fill='black')
L,R,T,Bm=80,950,70,370;axes(d,L,R,T,Bm,0,25,0,90,'max lift (mm)','final tilt (deg)')
for r in rows:
 if r['condition']!='D2':continue
 x=float(r['max_lift_m'])*1000;y=float(r['final_tilt_deg']);w=float(r['width_m'])*1000
 col={14:'#1f77b4',35:'#ff7f0e',40:'#d62728'}[int(w)]
 px,py=pt(min(25,x),min(90,y),L,R,T,Bm,0,25,0,90)
 d.ellipse([px-4,py-4,px+4,py+4],fill=col if r['success']=='True' else 'white',outline=col)
img.save(F/'figB2_lift_tilt.png')
# B3 failure modes
img=Image.new('RGB',(1000,450),'white');d=ImageDraw.Draw(img);d.text((30,12),'B3: failure modes (D2 only)',font=font(15),fill='black')
modes={'tilt>12':0,'gap>20mm':0,'|dy|>20mm':0,'success':0}
state_yaw={r['state_id']:float(r['obj_yaw_rad']) for r in csv.DictReader((B/'data/state_manifest.csv').open(encoding='utf-8-sig'))}
for r in rows:
 if r['condition']!='D2':continue
 if r['success']=='True':modes['success']+=1;continue
 if float(r['final_tilt_deg'])>12:modes['tilt>12']+=1
 if float(r['final_gap_m'])>0.02:modes['gap>20mm']+=1
 if abs(float(r['final_dy_m']))>0.02:modes['|dy|>20mm']+=1
 # no final yaw in csv; only report tilt/gap/dy
L,R,T,Bm=80,900,70,350;axes(d,L,R,T,Bm,0,4,0,8,'mode','count')
for i,(k,v) in enumerate(modes.items()):
 px,py=pt(i+0.5,v,L,R,T,Bm,0,4,0,8);d.rectangle([px-40,py,px+40,Bm],fill='#9467bd');d.text((px-5,py-14),str(v),font=font(9),fill='black');d.text((px-30,Bm+8),k,font=font(8),fill='black')
img.save(F/'figB3_failures.png')
print('done')
