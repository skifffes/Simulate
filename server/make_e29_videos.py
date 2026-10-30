import av
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
ROOT=Path('/scratch/lijh/simulate_vla'); B=ROOT/'results/experiment_2_9/ab'; O=B/'videos'; O.mkdir(parents=True,exist_ok=True)
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
def frames(p):
 c=av.open(str(p));out=[]
 for f in c.decode(video=0):out.append(Image.fromarray(f.to_ndarray(format='rgb24')).resize((640,480)))
 c.close();return out
def combine(a,b,out,la,lb,title):
 fa,fb=frames(a),frames(b);n=min(len(fa),len(fb));c=av.open(str(out),'w');st=c.add_stream('libx264',rate=20);st.width,st.height,st.pix_fmt=1300,540,'yuv420p'
 try:fnt=ImageFont.truetype(FONT,17);sm=ImageFont.truetype(FONT,15)
 except Exception:fnt=sm=ImageFont.load_default()
 for i in range(n):
  im=Image.new('RGB',(1300,540),'white');im.paste(fa[i],(5,55));im.paste(fb[i],(655,55));d=ImageDraw.Draw(im)
  d.text((10,12),title,font=fnt,fill='black');d.text((10,500),la,font=sm,fill='#1f77b4');d.text((660,500),lb,font=sm,fill='#d62728')
  for pkt in st.encode(av.VideoFrame.from_image(im)):c.mux(pkt)
 for pkt in st.encode():c.mux(pkt)
 c.close();print('wrote',out,n)
def side(kind,tag):
 inner=tag.rsplit('_r',1)[0]
 return B/'runs'/kind/tag/('expert' if kind=='E0' else 'expert_e1' if kind=='E1' else 'frozen')/inner/'side.mp4'
combine(side('E1','low_00_g035_r0'),side('D2','low_00_g035_r0'),O/'B_low00_g35_expert_vs_D2.mp4','E1 expert @35mm: success','D2 @35mm: fail (dy 32mm)','Low-start low_00: expert vs frozen D2 at 35mm')
combine(side('D2','low_00_g014_r0'),side('D2','low_00_g035_r0'),O/'B_low00_D2_g14_vs_g35.mp4','D2 @14mm: success','D2 @35mm: fail','Low-start low_00: D2 width 14 vs 35mm')
combine(side('E1','high_03_g035_r0'),side('D2','high_03_g035_r0'),O/'B_high03_g35_expert_vs_D2.mp4','E1 expert @35mm: success','D2 @35mm: fail (tilt 43.5deg)','High-start high_03: expert vs D2 at 35mm')
combine(side('E1','low_00_g040_r0'),side('D2','low_00_g040_r0'),O/'B_low00_g40_expert_vs_D2.mp4','E1 expert @40mm: success','D2 @40mm: fail (tilt 31.9deg)','Low-start low_00: expert vs D2 at 40mm')
