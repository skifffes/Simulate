import csv
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
B=Path(r'D:\python\simulate\results\experiment_2_11');D=B/'data';F=B/'figures';FONT='C:/Windows/Fonts/arial.ttf'
def font(s):
 try:return ImageFont.truetype(FONT,s)
 except:return ImageFont.load_default()
resp=list(csv.DictReader((D/'gripper_condition_response.csv').open(encoding='utf-8-sig')))
res=list(csv.DictReader((D/'condition_response_residual.csv').open(encoding='utf-8-sig')))
def axes(d,L,R,T,Bm,x0,x1,y0,y1,xl,yl):
 d.rectangle([L,T,R,Bm],outline='black')
 for i in range(5):
  y=Bm-(Bm-T)*i/4;d.line([L,y,R,y],fill=(225,225,225));d.text((8,y-6),f'{y0+(y1-y0)*i/4:.3g}',font=font(8),fill='black')
  x=L+(R-L)*i/4;d.text((x-12,Bm+6),f'{x0+(x1-x0)*i/4:.3g}',font=font(8),fill='black')
 d.text((L,Bm+20),xl,font=font(9),fill='black');d.text((L,T-12),yl,font=font(9),fill='black')
img=Image.new('RGB',(1000,450),'white');d=ImageDraw.Draw(img);d.text((30,12),'Figure 3: model vs expert EE z at h0 (initial observation)',font=font(13),fill='black')
axes(d,80,950,70,350,0,len(resp),40,100,'trajectory','EE z (mm)')
for i,r in enumerate(resp):
 for j,key in enumerate(['model_z_step0_mm','expert_z_step0_mm']):
  v=float(r[key]);x=80+(950-80)*(i+0.5)/len(resp)+(j-0.5)*12;y=350-(350-70)*(v-40)/60;d.rectangle([x-8,y,x+8,350],fill='#d62728' if j==0 else '#1f77b4')
img.save(F/'fig03_policy_vs_expert_ee_targets.png')
img=Image.new('RGB',(900,430),'white');d=ImageDraw.Draw(img);d.text((30,12),'Figure 9: model 35-14 vs 40-14 at initial observation',font=font(13),fill='black')
axes(d,80,850,70,330,0,4,0,40,'horizon','delta z (mm)')
for state,col in [('c3','#1f77b4'),('ep66','#d62728'),('c3_40','#2ca02c')]:
 sub=[r for r in res if r['state']==state];d.line([(80+(850-80)*(float(r['horizon'])/4),330-(330-70)*float(r['model_condition_diff_mm'])/40) for r in sub],fill=col,width=2)
img.save(F/'fig09_case_studies.png')
print('done')
