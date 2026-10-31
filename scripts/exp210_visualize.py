import csv
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(r'D:\python\simulate');B=ROOT/'results/experiment_2_10';D=B/'data';F=B/'figures';FONT='C:/Windows/Fonts/arial.ttf'
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
rows=pq.read_table(D/'b90_frame_state_action.parquet').to_pylist()
glob=list(csv.DictReader((D/'global_gripper_action_stats.csv').open(encoding='utf-8-sig')))
cov=list(csv.DictReader((D/'local_joint_coverage.csv').open(encoding='utf-8-sig')))
pr=list(csv.DictReader((D/'policy_response.csv').open(encoding='utf-8-sig')))
sup=list(csv.DictReader((D/'supported_policy_expert_comparison.csv').open(encoding='utf-8-sig')))
# fig1
img=Image.new('RGB',(1100,470),'white');d=ImageDraw.Draw(img);d.text((30,12),'Figure 1: B90 gripper opening across progress and inferred phase',font=font(15),fill='black')
L,R,T,Bm=80,1000,70,370;axes(d,L,R,T,Bm,0,1,0,38,'episode progress','grip (mm)')
cols={'initial':'#1f77b4','close_wait':'#2ca02c','descend':'#ff7f0e','push':'#d62728'}
for r in rows[::2]:
 px,py=pt(float(r['episode_progress']),float(r['gripper_per_finger_mm']),L,R,T,Bm,0,1,0,38);d.ellipse([px-1,py-1,px+1,py+1],fill=cols[r['phase_label']])
for i,(k,c) in enumerate(cols.items()):
 d.rectangle([80+i*160,420,92+i*160,432],fill=c);d.text((96+i*160,418),k,font=font(9),fill='black')
img.save(F/'fig01_gripper_and_phase.png')
# fig2 horizons
img=Image.new('RGB',(1000,460),'white');d=ImageDraw.Draw(img);d.text((30,12),'Figure 2: expert target height change by horizon',font=font(15),fill='black')
axes(d,80,950,70,360,0,3,0,500,'grip bin','mean delta z (mm)')
bins=[r for r in glob if r['bin']!='wide_34.5_35.5']+[r for r in glob if r['bin']=='wide_34.5_35.5']
for i,r in enumerate(bins):
 for j,h in enumerate([0,4,9,19,49]):
  v=-float(r[f'dz_h{h}_mean_mm']) if r[f'dz_h{h}_mean_mm']!='' else 0
  px,py=pt(i+0.4+j*0.12,max(0,v),80,950,70,360,0,3,0,500);d.rectangle([px-10,py,px+10,360],fill=['#1f77b4','#ff7f0e','#d62728','#9467bd','#2ca02c'][j])
 d.text((80+(950-80)*(i+0.5)/3-20,370),r['bin'].split('_')[0],font=font(8),fill='black')
img.save(F/'fig02_expert_targets_by_horizon.png')
# fig3 EE z vs grip
img=Image.new('RGB',(1000,460),'white');d=ImageDraw.Draw(img);d.text((30,12),'Figure 3: joint-state/gripper support (EE z vs grip)',font=font(15),fill='black')
axes(d,80,950,70,360,0,40,0,650,'grip (mm)','EE z (mm)')
for r in rows[::3]:
 px,py=pt(float(r['gripper_per_finger_mm']),min(650,float(r['ee_z'])*1000),80,950,70,360,0,40,0,650);d.ellipse([px-1,py-1,px+1,py+1],fill=(31,119,180))
img.save(F/'fig03_joint_state_gripper_support.png')
# fig4 C3/ep66 coverage bars at radius .75
img=Image.new('RGB',(1000,450),'white');d=ImageDraw.Draw(img);d.text((30,12),'Figure 4: C3 / ep66 local support (r=0.75 rad)',font=font(15),fill='black')
axes(d,80,950,70,350,0,3,0,1,'state','normalized count')
labels=['C3 (10)','ep66 f39'];vals=[(142,0),(1731,0)]
for i,(n,w) in enumerate(vals):
 px,py=pt(i+0.5,1,80,950,70,350,0,3,0,1);d.rectangle([px-60,py,px-10,350],fill='#1f77b4');d.rectangle([px+10,py,px+60,350],fill='#d62728')
 d.text((px-40,360),labels[i],font=font(9),fill='black')
img.save(F/'fig04_c3_ep66_neighbors.png')
# fig5 common support expert labels
img=Image.new('RGB',(1000,450),'white');d=ImageDraw.Draw(img);d.text((30,12),'Figure 5: local expert h0 labels (common support, not phase-matched)',font=font(13),fill='black')
axes(d,80,950,70,350,0,max(1,len(sup)),0,5,'state','expert dz h0 (mm)')
for i,r in enumerate(sup):
 px,py=pt(i+0.5,float(r['expert_dz_h0_narrow']),80,950,70,350,0,max(1,len(sup)),0,5);d.ellipse([px-3,py-3,px+3,py+3],fill='#1f77b4')
 py2=pt(i+0.5,float(r['expert_dz_h0_wide']),80,950,70,350,0,max(1,len(sup)),0,5)[1];d.ellipse([px-3,py2-3,px+3,py2+3],fill='#d62728')
img.save(F/'fig05_local_expert_labels.png')
# fig6 model response vs support
img=Image.new('RGB',(1000,450),'white');d=ImageDraw.Draw(img);d.text((30,12),'Figure 6: model 14->35 response vs wide local support',font=font(14),fill='black')
axes(d,80,950,70,350,0,20,0,70,'wide frames at r=.75','model delta z step10 (mm)')
for r in pr:
 wf=float(r['local_wide_frames']) if r['local_wide_frames']!='' else 0;v=float(r['model_delta_14_35_step10_mm'])
 px,py=pt(min(20,wf),v,80,950,70,350,0,20,0,70);d.ellipse([px-4,py-4,px+4,py+4],fill='#2ca02c')
img.save(F/'fig06_policy_response_by_support.png')
# fig7 identifiability
img=Image.new('RGB',(1100,500),'white');d=ImageDraw.Draw(img);d.text((30,12),'Figure 7: identifiability overview',font=font(16),fill='black')
ident=list(csv.DictReader((D/'identifiability_matrix.csv').open(encoding='utf-8-sig')))
for i,r in enumerate(ident):
 y=55+i*45;d.text((30,y),r['question'][:65],font=font(9),fill='black');d.text((700,y),r['status'][:40],font=font(9),fill='#d62728' if 'Not' in r['status'] else '#1f77b4')
img.save(F/'fig07_identifiability_overview.png')
print('done')
