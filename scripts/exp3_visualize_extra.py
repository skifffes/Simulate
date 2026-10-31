import csv
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
B=Path(r'D:\python\simulate\results\experiment_3');D=B/'data';F=B/'figures';FONT='C:/Windows/Fonts/arial.ttf'
def font(s):
 try:return ImageFont.truetype(FONT,s)
 except:return ImageFont.load_default()
cand=list(csv.DictReader((D/'candidate_pairs.csv').open(encoding='utf-8-sig')))
plan=list(csv.DictReader((D/'planning_results.csv').open(encoding='utf-8-sig')))
end=list(csv.DictReader((D/'endpoint_matching.csv').open(encoding='utf-8-sig')))
# fig1 state pair coverage
img=Image.new('RGB',(900,430),'white');d=ImageDraw.Draw(img);d.text((30,12),'A/B scene compatibility: object position difference',font=font(13),fill='black')
d.rectangle([80,70,850,330],outline='black')
for i,r in enumerate(cand):
 v=float(r['obj_pos_diff_mm']);x=80+(850-80)*(v/20);y=70+300*i/len(cand);d.ellipse([x-2,y-2,x+2,y+2],fill='#1f77b4')
d.text((80,340),'object position difference (mm); threshold 20mm',font=font(8),fill='black')
img.save(F/'fig01_state_pair_coverage.png')
# fig2 planning
img=Image.new('RGB',(900,430),'white');d=ImageDraw.Draw(img);d.text((30,12),'Pilot planning: q distance and collision-checked plan',font=font(13),fill='black')
d.rectangle([80,70,850,330],outline='black')
for i,r in enumerate(plan):
 v=float(r['scene_obj_diff_mm']);x=80+(850-80)*(i+0.5)/len(plan);h=(330-70)*min(1,v/20);d.rectangle([x-40,330-h,x+40,330],fill='#2ca02c')
 d.text((x-35,340),r['b_candidate_id'],font=font(8),fill='black')
img.save(F/'fig02_planning_success.png')
# fig6 object drift
img=Image.new('RGB',(900,430),'white');d=ImageDraw.Draw(img);d.text((30,12),'Object drift during transition (all ~1e-8 mm)',font=font(13),fill='black')
d.rectangle([80,70,850,330],outline='black')
for i,r in enumerate(end):
 v=max(1e-10,float(r['obj_drift_mm']));x=80+(850-80)*(i+0.5)/len(end);h=(330-70)*min(1,v/1e-6);d.rectangle([x-50,330-h,x+50,330],fill='#d62728')
img.save(F/'fig06_object_drift.png')
print('done')
