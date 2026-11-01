import csv
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
R=Path(r'D:\python\simulate\results\experiment_3_2');F=R/'figures';F.mkdir(exist_ok=True);FONT='C:/Windows/Fonts/arial.ttf'
def font(s):
 try:return ImageFont.truetype(FONT,s)
 except:return ImageFont.load_default()
rows=list(csv.DictReader((R/'b90_all_candidates.csv').open(encoding='utf-8-sig')))
sel=set(['ep068_f000','ep074_f000','ep079_f000','ep080_f000','ep082_f000','ep104_f000','ep064_f000'])
W,H=1000,520;img=Image.new('RGB',(W,H),'white');d=ImageDraw.Draw(img);d.text((20,10),'B90 start candidates: object distance vs joint distance',font=font(14),fill='black')
L,Rt,T,Bm=80,950,60,420;d.rectangle([L,T,Rt,Bm],outline='black')
for i in range(5):d.line([L+(Rt-L)*i/4,T,L+(Rt-L)*i/4,Bm],fill=(230,230,230));d.text((L+(Rt-L)*i/4-10,Bm+6),f'{i*5}',font=font(8),fill='black')
for i in range(5):d.text((20,Bm-(Bm-T)*i/4-5),f'{i*0.5:.1f}',font=font(8),fill='black')
for x in rows:
 x['scene_stratum']='Near' if float(x['obj_pos_diff_mm'])<=7.595385246558334 else ('Medium' if float(x['obj_pos_diff_mm'])<=11.615875968440838 else 'Far')
 px=L+(Rt-L)*float(x['obj_pos_diff_mm'])/20;py=Bm-(Bm-T)*float(x['q_dist_rad'])/2.0
 col='#2ca02c' if x['scene_stratum']=='Near' else ('#ff7f0e' if x['scene_stratum']=='Medium' else '#d62728')
 d.ellipse([px-3,py-3,px+3,py+3],fill=col)
 if x['b_candidate_id'] in sel:
  d.text((px+5,py-8),x['b_candidate_id'].replace('_f000',''),font=font(7),fill='black')
img.save(F/'candidate_overview.png')
print('ok')
