import csv
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
R=Path(r'D:\python\simulate\results\experiment_3_controlled');D=R/'data';S=R/'summaries';F=R/'figures';F.mkdir(exist_ok=True);FONT='C:/Windows/Fonts/arial.ttf'
def font(s):
 try:return ImageFont.truetype(FONT,s)
 except:return ImageFont.load_default()
g=list(csv.DictReader((S/'group_summary.csv').open(encoding='utf-8-sig')));p=list(csv.DictReader((S/'paired_seed.csv').open(encoding='utf-8-sig')));fa=list(csv.DictReader((S/'first_action_diff.csv').open(encoding='utf-8-sig')))
# fig1 alignment table
al=list(csv.DictReader((D/'state_alignment.csv').open(encoding='utf-8-sig')))
img=Image.new('RGB',(1000,500),'white');d=ImageDraw.Draw(img);d.text((20,10),'State alignment (A/B/C, ep064 f000)',font=font(15),fill='black')
sel=[x for x in al if x['pair']=='B_place_vs_C_transition' and x['metric'] in ('q_l2_rad','ee_pos_mm','obj_pos_mm','ee_obj_rel_mm','top_mae','wrist_mae','contacts_delta')]
for i,x in enumerate(sel):
 y=50+i*55;d.text((25,y),x['metric'],font=font(10),fill='black');d.text((380,y),f"{float(x['value']):.3g}",font=font(10),fill='black');d.text((560,y),f"threshold {x['threshold']}",font=font(9),fill='black');d.text((760,y),'PASS' if x['pass']=='True' else 'FAIL',font=font(10),fill='#2ca02c' if x['pass']=='True' else '#d62728')
img.save(F/'fig01_state_alignment.png')
# fig2 paired success/tilt
img=Image.new('RGB',(1000,450),'white');d=ImageDraw.Draw(img);d.text((20,10),'Paired seeds: success and final tilt',font=font(15),fill='black')
d.rectangle([70,60,950,330],outline='black')
for i,x in enumerate(p):
 xx=70+(950-70)*(i+0.5)/5
 for j,(gname,col) in enumerate([('A','#1f77b4'),('B','#d62728'),('C','#2ca02c')]):
  v=float(x[f'{gname}_tilt']);y=330-(330-60)*min(1,v/20);d.rectangle([xx-20+j*14,y,xx-8+j*14,330],fill=col)
 d.text((xx-25,340),f"seed {x['seed']}",font=font(8),fill='black')
img.save(F/'fig02_paired_outcomes.png')
# fig3 first action
img=Image.new('RGB',(900,430),'white');d=ImageDraw.Draw(img);d.text((20,10),'First action q L2 differences (paired seed)',font=font(13),fill='black')
d.rectangle([80,60,850,330],outline='black')
for i,x in enumerate(fa):
 xx=80+(850-80)*(i+0.5)/5
 for j,(k,c) in enumerate([('A_B_q_l2_rad','#1f77b4'),('B_C_q_l2_rad','#d62728')]):
  v=float(x[k]);y=330-(330-60)*min(1,v/0.01);d.rectangle([xx-25+j*25,y,xx-5+j*25,330],fill=c)
 d.text((xx-20,340),f"seed {x['seed']}",font=font(8),fill='black')
img.save(F/'fig03_first_action_diff.png')
# fig4 success/contact
img=Image.new('RGB',(900,430),'white');d=ImageDraw.Draw(img);d.text((20,10),'v2.2 success vs real wall contact (5 paired seeds/group)',font=font(12),fill='black')
d.rectangle([80,60,850,330],outline='black')
for i,x in enumerate(g):
 xx=80+(850-80)*(i+0.5)/3;v=int(x['v22_success']);y=330-(330-60)*v/5;d.rectangle([xx-60,y,xx-10,330],fill='#2ca02c')
 w=int(x['wall_contact']);y2=330-(330-60)*w/5;d.rectangle([xx+10,y2,xx+60,330],fill='#d62728')
 d.text((xx-45,340),x['group'],font=font(9),fill='black')
img.save(F/'fig04_success_contact.png')
# fig5 video contact sheet
frames=[Image.open(R/f'videos/frame_{n}.png').resize((640,260)) for n in ('start','transition','b90','end')]
canvas=Image.new('RGB',(1280,520),'white')
for i,fr in enumerate(frames):canvas.paste(fr,((i%2)*640,(i//2)*260))
canvas.save(F/'fig05_ato_b_video_frames.png')
print('ok')
