import csv
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
R=Path(r'D:\python\simulate\results\experiment_3_2');F=R/'figures';F.mkdir(exist_ok=True);FONT='C:/Windows/Fonts/arial.ttf'
def font(s):
 try:return ImageFont.truetype(FONT,s)
 except:return ImageFont.load_default()
g=list(csv.DictReader((R/'summary_by_target.csv').open(encoding='utf-8-sig')));s=list(csv.DictReader((R/'scene_effect.csv').open(encoding='utf-8-sig')))
img=Image.new('RGB',(1000,450),'white');d=ImageDraw.Draw(img);d.text((20,10),'A/B/C success by B target',font=font(14),fill='black');d.rectangle([70,60,950,330],outline='black')
targets=sorted(set(x['target'] for x in g))
for i,t in enumerate(targets):
 xx=70+(950-70)*(i+0.5)/len(targets)
 for j,(grp,col) in enumerate([('A','#1f77b4'),('B','#ff7f0e'),('C','#2ca02c')]):
  x=next(z for z in g if z['target']==t and z['group']==grp);v=int(x['success']);y=330-(330-60)*v/5;d.rectangle([xx-22+j*15,y,xx-10+j*15,330],fill=col)
 d.text((xx-28,340),t.replace('_f000',''),font=font(7),fill='black')
img.save(F/'summary_by_target.png')
img=Image.new('RGB',(900,430),'white');d=ImageDraw.Draw(img);d.text((20,10),'A/B success difference vs object shift',font=font(13),fill='black');d.rectangle([80,60,850,330],outline='black')
for x in s:
 xx=80+(850-80)*(float(x['AB_obj_mm'])/20);diff=int(x['A_success'])-int(x['B_success']);y=195-(diff*55);d.ellipse([xx-5,y-5,xx+5,y+5],fill='#2ca02c' if diff>=0 else '#d62728');d.text((xx+6,y-8),x['target'].replace('_f000',''),font=font(7),fill='black')
img.save(F/'scene_effect.png')
print('ok')
