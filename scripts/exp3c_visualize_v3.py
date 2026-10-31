import csv
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
R=Path(r'D:\python\simulate\results\experiment_3_controlled');D=R/'data';S=R/'summaries';F=R/'figures';FONT='C:/Windows/Fonts/arial.ttf'
def font(s):
 try:return ImageFont.truetype(FONT,s)
 except:return ImageFont.load_default()
g=list(csv.DictReader((S/'group_summary_v3.csv').open(encoding='utf-8-sig')));al=list(csv.DictReader((D/'state_alignment_v3.csv').open(encoding='utf-8-sig')))
img=Image.new('RGB',(1000,470),'white');d=ImageDraw.Draw(img);d.text((20,10),'V3 continuous B/C at first inference (all 5 seeds)',font=font(14),fill='black')
for i,(k,label) in enumerate([('q_l2_rad','q L2'),('ee_pos_mm','EE pos'),('rel_pos_mm','relative pos'),('rel_rot_deg','relative rot'),('grip_mm','grip'),('action7_l2','7-joint action'),('top_mae','top MAE')]):
 v=[float(x[k]) for x in al];y=45+i*50;d.text((20,y),label,font=font(10),fill='black');d.text((300,y),f'max {max(v):.3g}',font=font(10),fill='black');d.text((650,y),'PASS',font=font(10),fill='#2ca02c')
img.save(F/'fig01_state_alignment.png')
img=Image.new('RGB',(900,430),'white');d=ImageDraw.Draw(img);d.text((20,10),'V3 paired continuous success and wall contact',font=font(12),fill='black');d.rectangle([80,60,850,330],outline='black')
for i,x in enumerate(g):
 xx=80+(850-80)*(i+0.5)/len(g);s=int(x['success']);y=330-(330-60)*s/5;d.rectangle([xx-45,y,xx-10,330],fill='#2ca02c');w=int(x['wall']);y2=330-(330-60)*w/5;d.rectangle([xx+10,y2,xx+45,330],fill='#d62728');d.text((xx-10,340),x['group'],font=font(10),fill='black')
img.save(F/'fig02_paired_outcomes.png')
img=Image.new('RGB',(900,430),'white');d=ImageDraw.Draw(img);d.text((20,10),'V3 full 7-joint first action L2 (B vs C)',font=font(13),fill='black');d.rectangle([80,60,850,330],outline='black')
for i,x in enumerate(al):
 xx=80+(850-80)*(i+0.5)/len(al);v=float(x['action7_l2']);y=330;d.rectangle([xx-8,y-2,xx+8,330],fill='#1f77b4')
img.save(F/'fig03_first_action_diff.png')
img=Image.new('RGB',(900,430),'white');d=ImageDraw.Draw(img);d.text((20,10),'V3 target convergence: q error to qB',font=font(13),fill='black');d.rectangle([80,60,850,330],outline='black')
for i,x in enumerate(al):
 xx=80+(850-80)*(i+0.5)/len(al);v=float(x['q_l2_rad']);y=330-(330-60)*min(1,v/0.01);d.rectangle([xx-20,y,xx+20,330],fill='#9467bd')
img.save(F/'fig04_success_contact.png')
print('ok')
