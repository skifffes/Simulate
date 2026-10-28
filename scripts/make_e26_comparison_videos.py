import av
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/scratch/lijh/simulate_vla')
OUT = ROOT / 'results/experiment_2_6/videos'
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
    container = av.open(str(out), 'w')
    stream = container.add_stream('libx264', rate=20)
    stream.width, stream.height, stream.pix_fmt = 1300, 540, 'yuv420p'
    for i in range(n):
        canvas = Image.new('RGB', (1300, 540), 'white')
        canvas.paste(fa[i], (5, 55)); canvas.paste(fb[i], (655, 55))
        d = ImageDraw.Draw(canvas)
        try:
            f = ImageFont.truetype(FONT, 18)
        except Exception:
            f = ImageFont.load_default()
        d.text((10, 12), title, font=f, fill='black')
        d.text((10, 500), la, font=f, fill='#1f77b4')
        d.text((660, 500), lb, font=f, fill='#d62728')
        frame = av.VideoFrame.from_image(canvas)
        for pkt in stream.encode(frame):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()
    print('wrote', out, n)


combine(ROOT / 'results/experiment_2_6/A_midstate_launch/expert/ep022_f039/side.mp4',
        ROOT / 'results/experiment_2_6/A_midstate_launch/frozen/ep022_f039/side.mp4',
        OUT / 'A_expert_vs_frozen_D2.mp4', 'expert suffix: success', 'frozen D2: fail',
        'A: ep22 f39 expert vs frozen D2')
combine(ROOT / 'results/experiment_2_6/B_gripper_variation/expert/ep066_f039_g025/side.mp4',
        ROOT / 'results/experiment_2_6/B_gripper_variation/frozen/ep066_f039_g025/side.mp4',
        OUT / 'B_expert_vs_frozen_D2.mp4', 'expert: fail', 'frozen D2: success',
        'B: ep66 f39, 25 mm gripper')
combine(ROOT / 'results/experiment_2_6/B_gripper_variation/frozen/ep022_f039_g025/side.mp4',
        ROOT / 'results/experiment_2_6/B_gripper_variation/frozen/ep022_f039_g035/side.mp4',
        OUT / 'B_different_gripper_initializations.mp4', '25 mm: success', '35 mm: fail',
        'B: ep22 f39 frozen D2, 25 vs 35 mm')
combine(ROOT / 'results/experiment_2_5/G1/C3/seed_40000000/side.mp4',
        ROOT / 'results/experiment_2_6/A_midstate_launch/frozen/ep066_f039/side.mp4',
        OUT / 'C3_vs_D2_midstate.mp4', 'C3: fail', 'D2 midstate ep66 f39: success',
        'C3 vs D2 midstate launch')
