import av
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/scratch/lijh/simulate_vla')
OUT = ROOT / 'results/experiment_2_6/b90_extension_videos'
OUT.mkdir(parents=True, exist_ok=True)
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def frames(path):
    c = av.open(str(path)); out = []
    for f in c.decode(video=0):
        out.append(Image.fromarray(f.to_ndarray(format='rgb24')).resize((640, 480)))
    c.close(); return out


def combine(a, b, out, la, lb, title):
    fa, fb = frames(a), frames(b)
    n = min(len(fa), len(fb))
    c = av.open(str(out), 'w'); st = c.add_stream('libx264', rate=20)
    st.width, st.height, st.pix_fmt = 1300, 540, 'yuv420p'
    try:
        fnt = ImageFont.truetype(FONT, 18)
        small = ImageFont.truetype(FONT, 16)
    except Exception:
        fnt = ImageFont.load_default()
        small = ImageFont.load_default()
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


combine(ROOT / 'results/experiment_2_5/G1/C3/seed_40000000/side.mp4',
        ROOT / 'results/experiment_2_6/C3_direct_B90/frozen/seed_40000000/side.mp4',
        OUT / 'C3_original_vs_independent_B90.mp4', 'original C3: fail', 'independent B90: fail',
        'C3 original switching vs independent frozen B90')
combine(ROOT / 'results/experiment_2_6/C3_direct_B90/frozen/seed_40000000/side.mp4',
        ROOT / 'results/experiment_2_6/C3_direct_B90/expert/seed_40000000/side.mp4',
        OUT / 'C3_frozen_vs_expert_B90.mp4', 'frozen D2: fail', 'expert B90: success',
        'C3 terminal: frozen D2 vs closed-loop expert')
combine(ROOT / 'results/experiment_2_6/C3_direct_B90/frozen/seed_40000000/side.mp4',
        ROOT / 'results/experiment_2_6/B_gripper_variation/frozen/ep066_f039_g040/side.mp4',
        OUT / 'C3_vs_nearest_B90_midstate.mp4', 'C3 terminal: fail', 'ep66 f39 40mm (2.6): success',
        'C3 vs nearest B90 midstate')
combine(ROOT / 'results/experiment_2_6/B90_gripper_extension/frozen/ep082_f039_g025/side.mp4',
        ROOT / 'results/experiment_2_6/B90_gripper_extension/frozen/ep082_f039_g040/side.mp4',
        OUT / 'B90_gripper_width_comparison.mp4', '25mm: success', '40mm: fail',
        'B90 extension: ep82 f39 width comparison')
combine(ROOT / 'results/experiment_2_6/B90_gripper_extension/expert/ep082_f039_g035/side.mp4',
        ROOT / 'results/experiment_2_6/B90_gripper_extension/frozen/ep082_f039_g035/side.mp4',
        OUT / 'B90_expert_vs_frozen_D2.mp4', 'expert: success', 'frozen D2: fail',
        'B90 extension: ep82 f39 35mm expert vs frozen')
