import csv, json, math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(r'D:\python\simulate')
OUT = ROOT / 'results/experiment_2_5/d2_c3_coverage/figures'
OUT.mkdir(parents=True, exist_ok=True)
FONT = 'C:/Windows/Fonts/arial.ttf'


def font(s):
    try:
        return ImageFont.truetype(FONT, s)
    except Exception:
        return ImageFont.load_default()


def bars(path, labels, series, title, ylabel):
    W, H = 1000, 430; img = Image.new('RGB', (W, H), 'white'); d = ImageDraw.Draw(img)
    d.text((40, 18), title, font=font(18), fill='black')
    L, R, T, B = 80, 950, 70, 330; d.rectangle([L, T, R, B], outline='black')
    mx = max(max(v) for _, _, v in series) * 1.15 or 1
    slot = (R - L) / len(labels); bw = slot / (len(series) + 0.5)
    for i, lab in enumerate(labels):
        x0 = L + i * slot + slot * 0.08
        for j, (name, col, vals) in enumerate(series):
            h = (B - T) * vals[i] / mx; x = x0 + j * bw
            d.rectangle([x, B - h, x + bw * 0.88, B], fill=col, outline='black')
            d.text((x, B - h - 13), f'{vals[i]:.3f}', font=font(9), fill='black')
        d.text((x0, B + 8), lab, font=font(9), fill='black')
    for j, (name, col, _) in enumerate(series):
        x = L + j * 200; y = B + 42
        d.rectangle([x, y, x + 12, y + 12], fill=col, outline='black')
        d.text((x + 16, y - 2), name, font=font(10), fill='black')
    img.save(path)


def line(path, series, title, ylabel):
    W, H = 1000, 430; img = Image.new('RGB', (W, H), 'white'); d = ImageDraw.Draw(img)
    d.text((40, 18), title, font=font(17), fill='black')
    L, R, T, B = 80, 950, 70, 330; d.rectangle([L, T, R, B], outline='black')
    xs = [x for _, _, x, _ in series]; ys = [y for _, _, _, y in series]
    xmin, xmax = min(min(v) for v in xs), max(max(v) for v in xs)
    ymin, ymax = min(min(v) for v in ys), max(max(v) for v in ys)
    pad = (ymax - ymin) * 0.1 or 0.005; ymin -= pad; ymax += pad
    for i in range(5):
        y = B - (B - T) * i / 4
        d.line([L, y, R, y], fill=(225, 225, 225))
        d.text((10, y - 7), f'{ymin + (ymax - ymin) * i / 4:.3f}', font=font(9), fill='black')
    for i in range(5):
        x = L + (R - L) * i / 4
        d.text((x - 10, B + 8), f'{xmin + (xmax - xmin) * i / 4:.0f}', font=font(9), fill='black')
    for name, col, x, y in series:
        pts = [(L + (R - L) * (xi - xmin) / (xmax - xmin),
                B - (B - T) * (yi - ymin) / (ymax - ymin)) for xi, yi in zip(x, y)]
        d.line(pts, fill=col, width=2)
    for i, (name, col, _, _) in enumerate(series):
        y = B + 38 + i * 15
        d.line([L, y, L + 15, y], fill=col, width=3)
        d.text((L + 20, y - 7), name, font=font(10), fill='black')
    d.text((L, B + 12), 'control tick', font=font(9), fill='black')
    img.save(path)


summary = json.loads((ROOT / 'results/experiment_2_5/d2_c3_coverage/analysis_summary.json').read_text(encoding='utf-8'))
qs = summary['queries']
# Fig 2
bars(OUT / 'fig2_start_vs_interior.png', [str(q['seed']) for q in qs],
     [('start', '#1f77b4', [q['start_min'] for q in qs]),
      ('interior (all)', '#d62728', [q['interior_min'] for q in qs]),
      ('interior yaw90', '#2ca02c', [q['interior_yaw90_min'] for q in qs])],
     'Figure 2: C3 vs D2 B start / interior nearest (q L2)', 'rad')
# Fig 4 relative geometry
init = list(csv.DictReader(open(ROOT / 'results/experiment_2_5/d2_c3_coverage/initial_nn.csv', encoding='utf-8-sig')))
# use known physical values for seed0 start ep74 and ep66 from computation
c3 = [0.44975 - 0.54794, 0.18298 - 0.18289]
start = [0.44628486037254333 - 0.54628 - 0.15235, 0.1761242300271988 - 0.17611]
# recompute exactly from local D2 metadata for ep74
rows = [json.loads(x) for x in (ROOT / 'results/experiment_2_3/data/D2_episodes.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
ok = [r for r in rows if r['success'] and r['reason'] == 'ok']
ep74 = next(r for r in ok if r['index'] == 74)
start_rel = [float(ep74['start_ee'][0] - ep74['scene']['block_start'][0]),
             float(ep74['start_ee'][1] - ep74['scene']['block_start'][1])]
exp_rel = [-0.09999513962745665, 0.0000142300271988]
W, H = 1000, 430; img = Image.new('RGB', (W, H), 'white'); d = ImageDraw.Draw(img)
d.text((40, 18), 'Figure 4: relative EE-object position (x-z, m)', font=font(18), fill='black')
L, R, T, B = 250, 800, 70, 380; d.rectangle([L, T, R, B], outline='black')
pts = [('C3 (-0.098, 0.050)', (c3[0], 0.04958), '#d62728'),
       ('start ep74', (start_rel[0], float(ep74['start_ee'][2] - ep74['scene']['block_start'][2])), '#1f77b4'),
       ('interior ep66 f39 (-0.100, 0.050)', (exp_rel[0], 0.050045), '#2ca02c')]
for name, (px, pz), col in pts:
    x = L + (R - L) * (px + 0.115) / 0.025
    y = B - (B - T) * (pz - 0.035) / 0.030
    d.ellipse([x - 7, y - 7, x + 7, y + 7], fill=col)
    d.text((x + 10, y - 10), name, font=font(11), fill=col)
d.text((L, B + 10), 'relative x (m)', font=font(10), fill='black')
d.text((10, T - 20), 'relative z (m)', font=font(10), fill='black')
img.save(OUT / 'fig4_relative_geometry.png')
# Fig 5 expert vs VLA EE z
comp = list(csv.DictReader(open(ROOT / 'results/experiment_2_5/d2_c3_coverage/nearest_trajectory_comparison_yaw90.csv', encoding='utf-8-sig')))
sub = [r for r in comp if r['query_seed'] == '40000000']
line(OUT / 'fig5_nearest_subsequent.png',
     [('expert EE z (D2 ep66 f39+)', '#d62728', [int(r['t']) for r in sub], [float(r['expert_ee_z']) for r in sub]),
      ('C3 VLA EE z', '#1f77b4', [int(r['t']) for r in sub], [float(r['vla_ee_z']) for r in sub])],
     'Figure 5: expert ep66 f39 (obs grip 14 mm) vs C3 VLA (obs grip 40 mm)', 'EE z (m)')
# Fig 6 reference bars
ref = summary['reference']
bars(OUT / 'fig6_reference_distribution.png', ['start NN', 'interior episode-min'],
     [('median', '#1f77b4', [ref['start_median'], ref['interior_episode_min_median']]),
      ('p95', '#ff7f0e', [ref['start_p95'], ref['interior_episode_min_p95']]),
      ('max', '#d62728', [ref['start_max'], ref['interior_episode_min_max']])],
     'Figure 6: D2 internal reference distances (rad)', 'rad')
print('done')
bars(OUT / 'fig7_state_alignment.png',
     ['C3 obs', 'C3 action', 'D2 B start obs', 'D2 B start action', 'ep66 f39 obs', 'ep66 f39 action'],
     [('gripper (mm)', '#1f77b4', [40.0, 16.56, 35.0, 14.0, 14.0, 14.0])],
     'Figure 7: observation vs action gripper fields (mm)', 'mm')
