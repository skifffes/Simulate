import csv, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(r'D:\python\simulate')
SRC = ROOT / 'results/experiment_2_3/data/D2_episodes.jsonl'
OUT = ROOT / 'results/experiment_2_8/supplement_s1'
rows = [json.loads(x) for x in SRC.read_text(encoding='utf-8').splitlines() if x.strip()]
rows = [r for r in rows if r['success'] and r['reason'] == 'ok']
recs = []
for r in rows:
    ee = np.array(r['start_ee'], float)
    blk = np.array(r['scene']['block_start'], float)
    recs.append({'index': r['index'], 'group': r['spec']['group'], 'mode': r['mode'],
                 'home_distance_rad': r['home_distance_rad'],
                 'ee_x': ee[0], 'ee_y': ee[1], 'ee_z': ee[2],
                 'block_x': blk[0], 'block_y': blk[1], 'block_z': blk[2],
                 'ee_block_xy_m': float(np.linalg.norm(ee[:2] - blk[:2])),
                 'ee_z_above_block_m': float(ee[2] - blk[2]),
                 'block_yaw_deg': float(r['scene']['block_yaw'] * 180 / np.pi),
                 'gap_m': float(r['spec']['gap_target_m'])})
with (OUT / 'data/d2_start_distribution.csv').open('w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(recs[0].keys())); w.writeheader(); w.writerows(recs)

FONT = 'C:/Windows/Fonts/arial.ttf'
def font(s):
    try: return ImageFont.truetype(FONT, s)
    except Exception: return ImageFont.load_default()

colors = {'home': '#1f77b4', 'home_neighborhood': '#2ca02c',
          'multi_ik': '#ff7f0e', 'task_relevant': '#d62728'}
img = Image.new('RGB', (1200, 540), 'white'); d = ImageDraw.Draw(img)
d.text((30, 12), 'D2 B start distribution: end-effector at episode start', font=font(18), fill='black')
# xy
L, R, T, Bm = 70, 560, 70, 430
xmin, xmax, ymin, ymax = 0.30, 0.80, -0.15, 0.30
d.rectangle([L, T, R, Bm], outline='black')
for i in range(5):
    x = L + (R - L) * i / 4
    d.text((x - 12, Bm + 8), f'{xmin + (xmax - xmin) * i / 4:.2f}', font=font(9), fill='black')
    y = Bm - (Bm - T) * i / 4
    d.text((10, y - 6), f'{ymin + (ymax - ymin) * i / 4:.2f}', font=font(9), fill='black')
for r in recs:
    x = L + (R - L) * (r['ee_x'] - xmin) / (xmax - xmin)
    y = Bm - (Bm - T) * (r['ee_y'] - ymin) / (ymax - ymin)
    col = colors[r['mode']]
    if r['group'] == 'yaw90':
        d.rectangle([x - 3, y - 3, x + 3, y + 3], fill=col)
    else:
        d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=col)
# block centroid
bx = np.mean([r['block_x'] for r in recs]); by = np.mean([r['block_y'] for r in recs])
x = L + (R - L) * (bx - xmin) / (xmax - xmin); y = Bm - (Bm - T) * (by - ymin) / (ymax - ymin)
d.polygon([(x, y - 8), (x - 7, y + 6), (x + 7, y + 6)], outline='black', fill='#ffff00')
d.text((L, 450), 'x (m)', font=font(10), fill='black'); d.text((10, T - 18), 'y (m)', font=font(10), fill='black')
# xz
L2, R2, T2, Bm2 = 700, 1150, 70, 430
zmin, zmax = 0.0, 0.65
d.rectangle([L2, T2, R2, Bm2], outline='black')
for i in range(5):
    x = L2 + (R2 - L2) * i / 4
    d.text((x - 12, Bm2 + 8), f'{xmin + (xmax - xmin) * i / 4:.2f}', font=font(9), fill='black')
    y = Bm2 - (Bm2 - T2) * i / 4
    d.text((645, y - 6), f'{zmin + (zmax - zmin) * i / 4:.2f}', font=font(9), fill='black')
for r in recs:
    x = L2 + (R2 - L2) * (r['ee_x'] - xmin) / (xmax - xmin)
    y = Bm2 - (Bm2 - T2) * (r['ee_z'] - zmin) / (zmax - zmin)
    col = colors[r['mode']]
    if r['group'] == 'yaw90':
        d.rectangle([x - 3, y - 3, x + 3, y + 3], fill=col)
    else:
        d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=col)
d.text((L2, 450), 'x (m)', font=font(10), fill='black'); d.text((645, T2 - 18), 'z (m)', font=font(10), fill='black')
for i, (name, col) in enumerate(colors.items()):
    d.rectangle([70 + i * 180, 490, 82 + i * 180, 502], fill=col)
    d.text((86 + i * 180, 488), name, font=font(10), fill='black')
d.text((70, 515), 'circle=yaw0 start, square=yaw90 start; yellow triangle=block centroid; all 120 successful B starts', font=font(9), fill='black')
img.save(OUT / 'figures/fig09_d2_start_distribution.png')
print('done', len(recs))
