import av
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
ROOT=Path('/scratch/lijh/simulate_vla');B=ROOT/'results/experiment_2_9/b2_low_near';O=B/'videos';O.mkdir(parents=True,exist_ok=True)
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
 inner=tag.rsplit('_r',1)[0];sub='expert' if kind=='E0' else 'expert_e1' if kind=='E1' else 'frozen'
 return B/'runs'/kind/tag/sub/inner/'side.mp4'
combine(side('D2','low_near_00_g014_r0'),side('D2','low_near_00_g035_r0'),O/'B2_low_near00_D2_g14_vs_g35.mp4','D2 14mm: success','D2 35mm: success','True Low-Near low_near_00: D2 14 vs 35mm')
combine(side('E0','low_near_00_g035_r0'),side('D2','low_near_00_g035_r0'),O/'B2_low_near00_expert_vs_D2_g35.mp4','E0 expert 35mm: success','D2 35mm: success','Low-Near low_near_00: expert vs D2 at 35mm')
combine(side('D2','low_near_02_g014_r0'),side('D2','low_near_02_g040_r0'),O/'B2_low_near02_D2_g14_vs_g40.mp4','D2 14mm: success','D2 40mm: success','Low-Near low_near_02: D2 14 vs 40mm')
