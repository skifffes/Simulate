import csv
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(r'D:\python\simulate')
B = ROOT / 'results/experiment_2_7'
FIG = B / 'figures'; FIG.mkdir(parents=True, exist_ok=True)
FONT = 'C:/Windows/Fonts/arial.ttf'


def font(s):
    try:
        return ImageFont.truetype(FONT, s)
    except Exception:
        return ImageFont.load_default()


def line(path, series, title, xlabel, ylabel, events=None, xlim=None, ylim=None):
    W, H = 1000, 460; img = Image.new('RGB', (W, H), 'white'); d = ImageDraw.Draw(img)
    d.text((40, 18), title, font=font(17), fill='black')
    L, R, T, Bm = 80, 950, 70, 360
    d.rectangle([L, T, R, Bm], outline='black')
    xs = [x for _, _, x, _ in series]; ys = [y for _, _, _, y in series]
    xmin, xmax = (min(min(v) for v in xs), max(max(v) for v in xs)) if not xlim else xlim
    ymin, ymax = (min(min(v) for v in ys), max(max(v) for v in ys)) if not ylim else ylim
    pad = (ymax - ymin) * 0.1 or 0.01; ymin -= pad; ymax += pad
    for i in range(5):
        y = Bm - (Bm - T) * i / 4
        d.line([L, y, R, y], fill=(225, 225, 225))
        d.text((10, y - 7), f'{ymin + (ymax - ymin) * i / 4:.3f}', font=font(9), fill='black')
    for i in range(5):
        x = L + (R - L) * i / 4
        d.text((x - 10, Bm + 8), f'{xmin + (xmax - xmin) * i / 4:.1f}', font=font(9), fill='black')
    for name, col, x, y in series:
        pts = [(L + (R - L) * (xi - xmin) / (xmax - xmin),
                Bm - (Bm - T) * (yi - ymin) / (ymax - ymin)) for xi, yi in zip(x, y)]
        d.line(pts, fill=col, width=2)
    for i, (name, col, _, _) in enumerate(series):
        y = Bm + 35 + i * 15
        d.line([L, y, L + 15, y], fill=col, width=3)
        d.text((L + 20, y - 7), name, font=font(10), fill='black')
    d.text((L, Bm + 12), xlabel, font=font(9), fill='black')
    for name, (xe, col) in (events or {}).items():
        px = L + (R - L) * (xe - xmin) / (xmax - xmin)
        d.line([px, T, px, Bm], fill=col, width=2)
        d.text((px + 3, T + 2), name, font=font(9), fill=col)
    img.save(path)


rows = list(csv.DictReader((B / 'observation_intervention/action_statistics.csv').open(encoding='utf-8-sig')))
widths = [14, 20, 25, 30, 35, 40]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
series = []
for w, c in zip(widths, colors):
    r = next(x for x in rows if x['set'] == 'real40' and abs(float(x['width_m']) * 1000 - w) < 0.1)
    series.append((f'{w}mm', c, [0, 5, 10, 20, 49],
                   [float(r[f'ee_z_t{k}']) for k in (0, 5, 10, 20, 49)]))
line(FIG / 'fig1_ee_prediction.png', series, 'Figure 1: predicted EE z vs action step (real 40mm images)',
     'action step', 'EE z (m)')
line(FIG / 'fig2_joint_l2.png',
     [('joint L2 vs 14mm', '#d62728', widths, [float(next(x for x in rows if x['set'] == 'real40' and abs(float(x['width_m'])*1000-w)<0.1)['joint_l2_mean']) for w in widths])],
     'Figure 2: joint action difference vs 14mm (paired seeds)', 'gripper observation (mm)', 'L2 (rad)')

c3 = list(csv.DictReader((B / 'c3_ee_height.csv').open(encoding='utf-8-sig')))
for name, col in (('c3_14mm', '#2ca02c'), ('c3_40mm', '#d62728')):
    sub = [r for r in c3 if r['condition'] == name]
    series = [(name, col, [float(r['time_s']) for r in sub], [float(r['ee_z']) for r in sub])]
    if name == 'c3_14mm':
        s14 = series
    else:
        s40 = series
line(FIG / 'fig3_c3_ee_height.png', s14 + s40, 'Figure 3: C3 actual EE height before contact',
     'time (s)', 'EE z (m)', events={'14 contact': (0.746, '#2ca02c'), '40 contact': (0.613, '#d62728')})

cross = list(csv.DictReader((B / 'crossed_action_replay/contact_statistics.csv').open(encoding='utf-8-sig')))
labels = [r['condition'] for r in cross]
W, H = 1000, 430; img = Image.new('RGB', (W, H), 'white'); d = ImageDraw.Draw(img)
d.text((40, 18), 'Figure 4: cross-replay contact outcomes', font=font(17), fill='black')
L, R, T, Bm = 80, 950, 70, 330; d.rectangle([L, T, R, Bm], outline='black')
slot = (R - L) / len(cross)
for i, r in enumerate(cross):
    x0 = L + i * slot + slot * 0.1
    for j, (key, col) in enumerate([('success', '#2ca02c'), ('max_lift_m', '#d62728')]):
        v = float(r[key]) if key == 'max_lift_m' else float(r[key] == 'True')
        h = (Bm - T) * (min(1.0, v) if key == 'success' else min(1.0, v * 10))
        x = x0 + j * 35
        d.rectangle([x, Bm - h, x + 30, Bm], fill=col, outline='black')
        d.text((x, Bm - h - 12), ('1' if v else '0') if key == 'success' else f'{v:.2f}', font=font(8), fill='black')
    d.text((x0, Bm + 8), r['condition'], font=font(8), fill='black')
img.save(FIG / 'fig4_cross_replay.png')
print('done')
