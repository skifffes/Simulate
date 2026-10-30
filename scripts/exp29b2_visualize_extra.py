import csv
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
B=Path(r'D:\python\simulate\results\experiment_2_9\b2_low_near');F=B/'figures';FONT='C:/Windows/Fonts/arial.ttf'
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
tr=list(csv.DictReader((B/'data/actual_ee_trajectories.csv').open(encoding='utf-8-sig')))
out=list(csv.DictReader((B/'data/rollout_outcomes.csv').open(encoding='utf-8-sig')))
# fig7
img=Image.new('RGB',(1000,450),'white');d=ImageDraw.Draw(img);d.text((30,12),'Figure 7: low_near_00 D2 vs E0 expert at 35 mm',font=font(15),fill='black')
axes(d,80,950,70,350,0,2.5,0,180,'time (s)','EE z (mm)')
for cond,col in [('D2','#d62728'),('E0','#1f77b4')]:
 sub=[r for r in tr if r['state_id']=='low_near_00' and int(float(r['width_mm']))==35 and r['condition']==cond]
 d.line([pt(float(r['t']),float(r['ee_z'])*1000,80,950,70,350,0,2.5,0,180) for r in sub],fill=col,width=2)
img.save(F/'fig07_expert_vs_d2.png')
# fig8 contact/lift
img=Image.new('RGB',(1000,450),'white');d=ImageDraw.Draw(img);d.text((30,12),'Figure 8: contact height and max lift by width (low_near)',font=font(14),fill='black')
axes(d,80,950,70,350,0,3,0,60,'width (14/35/40)','mm')
for i,w in enumerate([14,35,40]):
 sub=[r for r in out if r['condition']=='D2' and 'low_near' in r['state_id'] and int(float(r['width_mm']))==w]
 cz=np.mean([float(r['contact_ee_z_mm']) for r in sub if r['contact_ee_z_mm']!='']) if sub else 0
 lf=np.mean([float(r['max_lift_mm']) for r in sub]) if sub else 0
 for j,v,col in [(0,cz,'#1f77b4'),(1,lf,'#d62728')]:
  px,py=pt(i+0.4+j*0.2,v,80,950,70,350,0,3,0,60);d.rectangle([px-18,py,px+18,350],fill=col);d.text((px-12,py-12),f'{v:.1f}',font=font(8),fill='black')
img.save(F/'fig08_contact_comparison.png')
# fig9 actual diff vs predicted diff low_near00
img=Image.new('RGB',(1000,450),'white');d=ImageDraw.Draw(img);d.text((30,12),'Figure 9: predicted vs actual 35-14 height difference (low_near_00)',font=font(13),fill='black')
axes(d,80,950,70,350,0,2.5,0,80,'time (s)','delta z (mm)')
# actual diff at t via interpolation
for rep,col in [('0','#d62728'),('1','#ff7f0e')]:
 subs={}
 for w in [14,35]:
  s=[r for r in tr if r['state_id']=='low_near_00' and r['condition']=='D2' and int(float(r['width_mm']))==w and r['rep']==rep]
  subs[w]=s
 if not subs[14] or not subs[35]:continue
 t=np.array([float(r['t']) for r in subs[14]]);z14=np.array([float(r['ee_z']) for r in subs[14]])*1000
 t35=np.array([float(r['t']) for r in subs[35]]);z35=np.array([float(r['ee_z']) for r in subs[35]])*1000
 ts=np.linspace(0,2.5,60);d.line([pt(x,np.interp(x,t35,z35)-np.interp(x,t,z14),80,950,70,350,0,2.5,0,80) for x in ts],fill=col,width=2)
img.save(F/'fig09_prediction_convergence.png')
print('done')
