import csv
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
B=Path(r'D:\python\simulate\results\experiment_2_11');D=B/'data';F=B/'figures';F.mkdir(parents=True,exist_ok=True);FONT='C:/Windows/Fonts/arial.ttf'
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
meta=list(csv.DictReader((D/'observation_meta.csv').open(encoding='utf-8-sig')))
resp=list(csv.DictReader((D/'gripper_condition_response.csv').open(encoding='utf-8-sig')))
res=list(csv.DictReader((D/'condition_response_residual.csv').open(encoding='utf-8-sig')))
ph=list(csv.DictReader((D/'phase_level_summary.csv').open(encoding='utf-8-sig')))
# fig1 timelines
img=Image.new('RGB',(1100,470),'white');d=ImageDraw.Draw(img);d.text((30,12),'Figure 1: expert gripper/arm timeline (C3)',font=font(15),fill='black')
axes(d,80,1000,70,360,0,4.5,0,180,'time (s)','EE z (mm)')
for tr,col in [('c3_g014','#1f77b4'),('c3_g035','#d62728')]:
 sub=[r for r in meta if r['trajectory']==tr];d.line([pt(float(r['time_s']),float(r['ee_z_mm']),80,1000,70,360,0,4.5,0,180) for r in sub],fill=col,width=2)
img.save(F/'fig01_expert_gripper_arm_timeline.png')
# fig2 initial model vs expert z
img=Image.new('RGB',(1000,450),'white');d=ImageDraw.Draw(img);d.text((30,12),'Figure 2: initial-step EE target (model vs expert)',font=font(14),fill='black')
axes(d,80,950,70,350,0,len(resp),40,100,'trajectory','EE z (mm)')
for i,r in enumerate(resp):
 for j,(key,col) in enumerate([('model_z_step0_mm','#d62728'),('expert_z_step0_mm','#1f77b4')]):
  px,py=pt(i+0.5+j*0.2-0.1,float(r[key]),80,950,70,350,0,len(resp),40,100);d.rectangle([px-12,py,px+12,350],fill=col)
img.save(F/'fig02_policy_vs_expert_ee_targets.png')
# fig3 conditional response
img=Image.new('RGB',(1000,450),'white');d=ImageDraw.Draw(img);d.text((30,12),'Figure 3: model 35-14 vs expert 35-14 (initial observation)',font=font(14),fill='black')
axes(d,80,950,70,350,0,4,0,40,'horizon (policy steps)','delta z (mm)')
for state,col in [('c3','#1f77b4'),('ep66','#d62728')]:
 sub=[r for r in res if r['state']==state];d.line([pt(float(r['horizon']),float(r['model_condition_diff_mm']),80,950,70,350,0,4,0,40) for r in sub],fill=col,width=2)
img.save(F/'fig04_gripper_conditional_response.png')
# fig4 residual
img=Image.new('RGB',(1000,450),'white');d=ImageDraw.Draw(img);d.text((30,12),'Figure 4: condition response residual (model - expert)',font=font(14),fill='black')
axes(d,80,950,70,350,0,4,-5,40,'horizon','residual (mm)')
for state,col in [('c3','#1f77b4'),('ep66','#d62728'),('c3_40','#2ca02c')]:
 sub=[r for r in res if r['state']==state];d.line([pt(float(r['horizon']),float(r['residual_mm']),80,950,70,350,0,4,-5,40) for r in sub],fill=col,width=2)
img.save(F/'fig05_condition_response_residual.png')
# fig5 phase error
img=Image.new('RGB',(1000,450),'white');d=ImageDraw.Draw(img);d.text((30,12),'Figure 5: EE z action error by inferred phase',font=font(14),fill='black')
axes(d,80,950,70,350,0,3,0,20,'phase (initial/descend/push)','mean |z error| (mm)')
for i,p in enumerate(['initial','descend','push']):
 v=[abs(float(r['ee_z_err_mean_mm'])) for r in ph if r['phase']==p and r['horizon']=='0'];px,py=pt(i+0.5,np.mean(v) if v else 0,80,950,70,350,0,3,0,20);d.rectangle([px-40,py,px+40,350],fill='#9467bd')
img.save(F/'fig06_action_error_by_phase.png')
# fig6 C3 case steps
img=Image.new('RGB',(1000,450),'white');d=ImageDraw.Draw(img);d.text((30,12),'Figure 6: C3 model vs expert targets at initial observation',font=font(14),fill='black')
axes(d,80,950,70,350,0,4,0,110,'horizon (0,4,9,19)','EE z (mm)')
for w,col in [(14,'#1f77b4'),(35,'#d62728'),(40,'#2ca02c')]:
 r=next(x for x in resp if x['trajectory']==f'c3_g{w:03d}');d.line([pt(k,float(r[f'model_z_step{k}_mm']),80,950,70,350,0,4,0,110) for k in [0,4,9,19]],fill=col,width=2)
img.save(F/'fig07_c3_initial_action_comparison.png')
# fig7 support
img=Image.new('RGB',(900,430),'white');d=ImageDraw.Draw(img);d.text((30,12),'Figure 7: local wide 35mm support at C3/ep66',font=font(14),fill='black')
axes(d,80,850,70,330,0,2,0,1.1,'state','frames (wide=0)')
for i,(lab,v) in enumerate([('C3',0),('ep66 f39',0)]):
 px,py=pt(i+0.5,v,80,850,70,330,0,2,0,1.1);d.rectangle([px-40,py,px+40,330],fill='#d62728');d.text((px-30,340),lab,font=font(9),fill='black')
img.save(F/'fig08_training_support_vs_action_error.png')
print('done')
