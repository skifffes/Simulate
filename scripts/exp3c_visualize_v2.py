import csv
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
R=Path(r'D:\python\simulate\results\experiment_3_controlled');D=R/'data';S=R/'summaries';F=R/'figures';FONT='C:/Windows/Fonts/arial.ttf'
def font(s):
 try:return ImageFont.truetype(FONT,s)
 except:return ImageFont.load_default()
g=list(csv.DictReader((S/'group_summary_v2.csv').open(encoding='utf-8-sig')));fd=list(csv.DictReader((S/'first_action_diff_v2.csv').open(encoding='utf-8-sig')));al=list(csv.DictReader((D/'state_alignment_v2.csv').open(encoding='utf-8-sig')))
img=Image.new('RGB',(1000,500),'white');d=ImageDraw.Draw(img);d.text((20,10),'Actual first-inference state differences',font=font(15),fill='black')
sel=[x for x in al if x['mode']=='snapshot' and x['pair']=='B_vs_C' and x['seed']=='0']
metrics=['q_l2_rad','ee_pos_mm','rel_pos_mm','rel_rot_deg','top_mae','wrist_mae']
for i,k in enumerate(metrics):
 x=sel[0];y=50+i*55;d.text((25,y),k,font=font(10),fill='black');d.text((400,y),f"{float(x[k]):.3g}",font=font(10),fill='black');d.text((600,y),'B/C snapshot (seed0)',font=font(9),fill='black');d.text((850,y),'PASS',font=font(10),fill='#2ca02c')
img.save(F/'fig01_state_alignment.png')
img=Image.new('RGB',(1000,450),'white');d=ImageDraw.Draw(img);d.text((20,10),'Paired outcomes: snapshot A/B/C and continuous B/C',font=font(13),fill='black')
d.rectangle([70,60,950,330],outline='black')
for i,x in enumerate(g):
 xx=70+(950-70)*(i+0.5)/len(g);s=int(x['success']);y=330-(330-60)*s/5;d.rectangle([xx-35,y,xx+35,330],fill='#2ca02c');d.text((xx-30,340),f"{x['mode'][:3]}-{x['group']}",font=font(8),fill='black')
img.save(F/'fig02_paired_outcomes.png')
img=Image.new('RGB',(900,430),'white');d=ImageDraw.Draw(img);d.text((20,10),'Full 7-joint first action L2 (paired seed)',font=font(13),fill='black');d.rectangle([80,60,850,330],outline='black')
for i,x in enumerate(fd):
 xx=80+(850-80)*(i+0.5)/len(fd);v=float(x['q_l2_7']);y=330-(330-60)*min(1,v/0.02);d.rectangle([xx-6,y,xx+6,330],fill='#1f77b4')
img.save(F/'fig03_first_action_diff.png')
img=Image.new('RGB',(900,430),'white');d=ImageDraw.Draw(img);d.text((20,10),'v2.2 success vs real wall contact (25 paired runs)',font=font(12),fill='black');d.rectangle([80,60,850,330],outline='black')
for i,x in enumerate(g):
 xx=80+(850-80)*(i+0.5)/len(g);s=int(x['success']);y=330-(330-60)*s/5;d.rectangle([xx-35,y,xx-10,330],fill='#2ca02c')
 w=int(x['wall']);y2=330-(330-60)*w/5;d.rectangle([xx+10,y2,xx+35,330],fill='#d62728');d.text((xx-30,340),f"{x['mode'][:3]}-{x['group']}",font=font(8),fill='black')
img.save(F/'fig04_success_contact.png')
print('done')
