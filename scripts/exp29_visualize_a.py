import csv
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
ROOT=Path(r'D:\python\simulate'); B=ROOT/'results/experiment_2_9/ab'; F=B/'figures'; F.mkdir(parents=True,exist_ok=True)
D=B/'policy'; FONT='C:/Windows/Fonts/arial.ttf'
def font(s):
 try:return ImageFont.truetype(FONT,s)
 except:return ImageFont.load_default()
def axes(d,L,R,T,Bm,x0,x1,y0,y1,xl,yl):
 d.rectangle([L,T,R,Bm],outline='black')
 for i in range(5):
  y=Bm-(Bm-T)*i/4;d.line([L,y,R,y],fill=(225,225,225));d.text((8,y-6),f'{y0+(y1-y0)*i/4:.3g}',font=font(8),fill='black')
  x=L+(R-L)*i/4;d.text((x-10,Bm+6),f'{x0+(x1-x0)*i/4:.3g}',font=font(8),fill='black')
 d.text((L,Bm+20),xl,font=font(9),fill='black');d.text((L,T-12),yl,font=font(9),fill='black')
def pt(x,y,L,R,T,Bm,x0,x1,y0,y1):return (L+(R-L)*(x-x0)/(x1-x0),Bm-(Bm-T)*(y-y0)/(y1-y0))
mod=list(csv.DictReader((D/'modality_interventions.csv').open(encoding='utf-8-sig')))
sens=list(csv.DictReader((D/'policy_height_sensitivity.csv').open(encoding='utf-8-sig')))
colors={'P14-V14':'#1f77b4','P35-V35':'#ff7f0e','P40-V40':'#d62728','P14-V40':'#9467bd','P40-V14':'#2ca02c'}
# fig1
img=Image.new('RGB',(1200,470),'white');d=ImageDraw.Draw(img);d.text((30,12),'A1: physical-consistent predicted EE height (mean over 10 seeds)',font=font(16),fill='black')
for pi,kind in enumerate(['low','high']):
 L=60+pi*600;R=L+520;T=70;Bm=370;axes(d,L,R,T,Bm,0,49,0,420,'chunk step','EE z (mm)')
 for c in ['P14-V14','P35-V35','P40-V40']:
  vals=[next(float(r['mean_z_mm']) for r in mod if r['kind']==kind and r['condition']==c and int(r['chunk_step'])==k) for k in [0,4,9,19,49]]
  d.line([pt(k,v,L,R,T,Bm,0,49,0,420) for k,v in zip([0,4,9,19,49],vals)],fill=colors[c],width=2)
 d.text((L,T+4),kind.upper()+' start',font=font(11),fill='black')
for i,c in enumerate(['P14-V14','P35-V35','P40-V40']):
 d.rectangle([70+i*170,420,82+i*170,432],fill=colors[c]);d.text((86+i*170,418),c,font=font(10),fill='black')
img.save(F/'figA1_height_curves.png')
# fig2 bars at step9
img=Image.new('RGB',(1100,470),'white');d=ImageDraw.Draw(img);d.text((30,12),'A2: body vs image intervention at chunk step 10',font=font(16),fill='black')
L,R,T,Bm=80,1000,70,370;axes(d,L,R,T,Bm,0,4.3,0,350,'condition','EE z (mm)')
conds=['P14-V14','P14-V40','P40-V14','P40-V40']
for gi,kind in enumerate(['low','high']):
 for i,c in enumerate(conds):
  v=np.mean([float(r['mean_z_mm']) for r in mod if r['kind']==kind and r['condition']==c and int(r['chunk_step'])==9])
  px,py=pt(i+0.5,v,L,R,T,Bm,0,4.3,0,350)
  d.rectangle([px-30+gi*10-10,py,px+30+gi*10-10,Bm],fill='#1f77b4' if gi==0 else '#d62728')
  d.text((px-25+gi*10-10,py-12),f'{v:.0f}',font=font(8),fill='black')
for i,c in enumerate(conds): d.text((L+(R-L)*(i+0.5)/4.3-20,Bm+8),c,font=font(8),fill='black')
d.rectangle([80,410,92,422],fill='#1f77b4');d.text((96,408),'low',font=font(10),fill='black')
d.rectangle([180,410,192,422],fill='#d62728');d.text((196,408),'high',font=font(10),fill='black')
img.save(F/'figA2_modality.png')
# fig3 sensitivity
img=Image.new('RGB',(1000,450),'white');d=ImageDraw.Draw(img);d.text((30,12),'A3: physical-consistent 14->35 / 35->40 / 14->40 sensitivity',font=font(15),fill='black')
L,R,T,Bm=80,950,70,360;axes(d,L,R,T,Bm,0,4,0,80,'interval','mean dz at step10 (mm)')
ivs=['14-35','35-40','14-40']
for gi,kind in enumerate(['low','high']):
 for i,iv in enumerate(ivs):
  vals=[float(r['mean_delta_z_mm']) for r in sens if r['kind']==kind and r['intervention']==iv and int(r['chunk_step'])==9]
  v=np.mean(vals);px,py=pt(i+0.5,v,L,R,T,Bm,0,4,0,80)
  d.rectangle([px-35+gi*20-10,py,px+35+gi*20-10,Bm],fill='#1f77b4' if gi==0 else '#d62728')
  d.text((px-25+gi*20-10,py-14),f'{v:.1f}',font=font(9),fill='black')
for i,iv in enumerate(ivs):d.text((L+(R-L)*(i+0.5)/3-20,Bm+8),iv,font=font(9),fill='black')
d.rectangle([80,400,92,412],fill='#1f77b4');d.text((96,398),'low',font=font(10),fill='black')
d.rectangle([180,400,192,412],fill='#d62728');d.text((196,398),'high',font=font(10),fill='black')
img.save(F/'figA3_sensitivity.png')
print('done')
