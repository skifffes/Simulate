import av
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
ROOT = Path('/scratch/lijh/simulate_vla')
OUT = ROOT / 'results/experiment_2_7/videos'; OUT.mkdir(parents=True, exist_ok=True)
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'

def frames(p):
    c = av.open(str(p)); out = []
    for f in c.decode(video=0):
        out.append(Image.fromarray(f.to_ndarray(format='rgb24')).resize((640, 480)))
    c.close(); return out

def combine(a, b, out, la, lb, title):
    fa, fb = frames(a), frames(b); n = min(len(fa), len(fb))
    c = av.open(str(out), 'w'); st = c.add_stream('libx264', rate=20)
    st.width, st.height, st.pix_fmt = 1300, 540, 'yuv420p'
    try:
        fnt = ImageFont.truetype(FONT, 18); small = ImageFont.truetype(FONT, 16)
    except Exception:
        fnt = small = ImageFont.load_default()
    for i in range(n):
        canvas = Image.new('RGB', (1300, 540), 'white')
        canvas.paste(fa[i], (5, 55)); canvas.paste(fb[i], (655, 55))
        d = ImageDraw.Draw(canvas)
        d.text((10, 12), title, font=fnt, fill='black')
        d.text((10, 500), la, font=small, fill='#1f77b4')
        d.text((660, 500), lb, font=small, fill='#d62728')
        for pkt in st.encode(av.VideoFrame.from_image(canvas)): c.mux(pkt)
    for pkt in st.encode(): c.mux(pkt)
    c.close(); print('wrote', out, n)

combine(ROOT/'results/experiment_2_6/C3_seed40000000_frozen_14mm_x10/run00/frozen/seed_40000000_g014/side.mp4',
        ROOT/'results/experiment_2_6/C3_direct_B90/frozen/seed_40000000/side.mp4',
        OUT/'C3_14mm_vs_40mm.mp4', 'C3 14mm: success, side contact', 'C3 40mm: fail, top contact',
        'C3 initial gripper 14 vs 40 mm (same snapshot)')
combine(ROOT/'results/experiment_2_6/C3_seed40000000_frozen_14mm_x10/run00/frozen/seed_40000000_g014/side.mp4',
        ROOT/'results/experiment_2_6/C3_direct_B90/frozen/seed_40000000/side.mp4',
        OUT/'C3_first_contact_comparison.mp4', '14mm contact @0.746s', '40mm contact @0.613s',
        'C3 first contact comparison')
combine(ROOT/'results/experiment_2_7/crossed_action_replay/P14_g040/side.mp4',
        ROOT/'results/experiment_2_7/crossed_action_replay/P40_g014/side.mp4',
        OUT/'cross_replay_P14_vs_P40.mp4', 'P14 arm path + 40mm: success', 'P40 arm path + 14mm: fail',
        'Cross replay: arm path effect')
combine(ROOT/'results/experiment_2_7/crossed_action_replay/P14_g014/side.mp4',
        ROOT/'results/experiment_2_7/crossed_action_replay/P14_g040/side.mp4',
        OUT/'cross_replay_gripper_comparison.mp4', 'P14 + 14mm: success', 'P14 + 40mm: success',
        'Cross replay: initial gripper under same P14 path')
