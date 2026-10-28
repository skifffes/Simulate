import csv, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(r'D:\python\simulate')
OUT = ROOT / 'results/experiment_2_5/d2_c3_coverage/figures/fig3_joint_comparison.png'
rows = [json.loads(x) for x in (ROOT / 'results/experiment_2_3/data/D2_episodes.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
ok = [r for r in rows if r['success'] and r['reason'] == 'ok']
start = np.array(next(r for r in ok if r['index'] == 74)['start_q'])
comp = list(csv.DictReader(open(ROOT / 'results/experiment_2_5/d2_c3_coverage/nearest_trajectory_comparison_yaw90.csv', encoding='utf-8-sig')))
r0 = comp[0]
inter = np.array([float(r0[f'expert_q{i}']) for i in range(1, 8)])
c3 = np.array([0.70472, 0.33588, -0.28329, -2.43388, 0.24138, 2.74713, 0.99882])
vals = np.array([c3, start, inter])
W, H = 1000, 420
img = Image.new('RGB', (W, H), 'white'); d = ImageDraw.Draw(img)
d.text((40, 20), 'Figure 3: C3 vs nearest start (ep74) and interior (ep66 f39)', font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 17), fill='black')
L, R, T, B = 70, 960, 70, 330
d.rectangle([L, T, R, B], outline='black')
lo, hi = vals.min() - 0.2, vals.max() + 0.2
for j in range(7):
    x = L + 40 + j * 120
    for k, col in enumerate(['#1f77b4', '#d62728', '#2ca02c']):
        y = B - (B - T) * (vals[k, j] - lo) / (hi - lo)
        d.rectangle([x + k * 16, y, x + 14 + k * 16, B], fill=col, outline='black')
    d.text((x + 10, B + 8), f'q{j+1}', font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 10), fill='black')
for k, (col, lab) in enumerate([('#1f77b4', 'C3'), ('#d62728', 'start ep74'), ('#2ca02c', 'interior ep66 f39')]):
    y = B + 40
    x = L + k * 260
    d.rectangle([x, y, x + 14, y + 12], fill=col)
    d.text((x + 18, y - 2), lab, font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 11), fill='black')
img.save(OUT)
print('saved')
