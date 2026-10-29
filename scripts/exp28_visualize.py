import csv, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(r'D:\python\simulate')
B = ROOT / 'results/experiment_2_8'; FIG = B / 'figures'; FIG.mkdir(parents=True, exist_ok=True)
FONT = 'C:/Windows/Fonts/arial.ttf'


def font(s):
    try:
        return ImageFont.truetype(FONT, s)
    except Exception:
        return ImageFont.load_default()


def axes(d, L, R, T, Bm, xmin, xmax, ymin, ymax, xlab, ylab):
    d.rectangle([L, T, R, Bm], outline='black')
    for i in range(5):
        y = Bm - (Bm - T) * i / 4
        d.line([L, y, R, y], fill=(225, 225, 225))
        d.text((10, y - 6), f'{ymin + (ymax - ymin) * i / 4:.3g}', font=font(8), fill='black')
    for i in range(5):
        x = L + (R - L) * i / 4
        d.text((x - 10, Bm + 6), f'{xmin + (xmax - xmin) * i / 4:.3g}', font=font(8), fill='black')
    d.text((L, Bm + 20), xlab, font=font(9), fill='black')
    d.text((L, T - 14), ylab, font=font(9), fill='black')


def pt(x, y, L, R, T, Bm, xmin, xmax, ymin, ymax):
    return (L + (R - L) * (x - xmin) / (xmax - xmin),
            Bm - (Bm - T) * (y - ymin) / (ymax - ymin))


# Fig1
states = ['ep082_f010', 'ep080_f025', 'ep080_f055']
labels = ['ep082 f10 (early, max)', 'ep080 f25 (approach, min)', 'ep080 f55 (push, mid)']
W, H = 1200, 430; img = Image.new('RGB', (W, H), 'white'); d = ImageDraw.Draw(img)
d.text((30, 12), 'Figure 1: predicted EE height vs action step (mean over 10 seeds)', font=font(17), fill='black')
for si, (sid, lab) in enumerate(zip(states, labels)):
    z = np.load(B / 'policy' / f'{sid}.npz')['z']  # (width, seed, step)
    L = 60 + si * 380; R = L + 330; T = 60; Bm = 330
    for wi, w in enumerate((0, 2, 5)):
        y = z[wi].mean(0) * 1000
        col = ['#1f77b4', '#ff7f0e', '#d62728'][wi]
        pts = [pt(k, y[k], L, R, T, Bm, 0, 49, 0, 220) for k in range(50)]
        d.line(pts, fill=col, width=2)
    axes(d, L, R, T, Bm, 0, 49, 0, 220, 'chunk step', 'EE z (mm)')
    d.text((L + 5, T + 5), lab, font=font(10), fill='black')
for wi, w in enumerate((14, 25, 40)):
    d.line([60 + wi * 150, 393, 75 + wi * 150, 393], fill=['#1f77b4', '#ff7f0e', '#d62728'][wi], width=3)
    d.text((80 + wi * 150, 385), f'{w} mm', font=font(10), fill='black')
img.save(FIG / 'fig01_height_curves.png')

# data
sens = list(csv.DictReader((B / 'data/policy_sensitivity.csv').open(encoding='utf-8-sig')))
cross = list(csv.DictReader((B / 'data/cross_state_analysis.csv').open(encoding='utf-8-sig')))
# Fig2 heatmap
states = [r['state_id'] for r in sorted(cross, key=lambda r: (r['stage'], float(r['progress'])))]
ivs = ['14-25mm', '25-35mm', '35-40mm', '14-40mm']
table = {(r['state_id'], r['intervention']): float(r['mean_delta_z_mm']) for r in sens if int(r['chunk_step']) == 9}
W, H = 1100, 820; img = Image.new('RGB', (W, H), 'white'); d = ImageDraw.Draw(img)
d.text((30, 12), 'Figure 2: height sensitivity at chunk step 10 (mean delta z, mm)', font=font(16), fill='black')
L, T, cellw, cellh = 190, 60, 160, 19
vals = [table.get((s, iv), 0) for s in states for iv in ivs]
vmin, vmax = min(vals), max(vals)
for j, iv in enumerate(ivs):
    d.text((L + j * cellw + 5, T - 18), iv, font=font(11), fill='black')
for i, s in enumerate(states):
    d.text((10, T + i * cellh), s, font=font(8), fill='black')
    for j, iv in enumerate(ivs):
        v = table.get((s, iv), 0)
        t = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5
        col = (int(255 * (1 - t)), 80, int(255 * t))
        d.rectangle([L + j * cellw, T + i * cellh, L + (j + 1) * cellw - 2, T + (i + 1) * cellh - 2], fill=col)
        d.text((L + j * cellw + 5, T + i * cellh + 2), f'{v:.1f}', font=font(8), fill='black')
d.text((L, H - 25), f'color: blue={vmin:.1f} mm, red={vmax:.1f} mm', font=font(9), fill='black')
img.save(FIG / 'fig02_sensitivity_heatmap.png')

# Fig3 sensitivity vs progress
W, H = 1000, 450; img = Image.new('RGB', (W, H), 'white'); d = ImageDraw.Draw(img)
d.text((30, 12), 'Figure 3: 14->40 mm sensitivity vs episode progress', font=font(16), fill='black')
L, R, T, Bm = 80, 950, 70, 370
axes(d, L, R, T, Bm, 0, 1, 0, 70, 'episode progress', 'delta z at step10 (mm)')
cols = {'early': '#1f77b4', 'approach': '#2ca02c', 'precontact': '#ff7f0e', 'push': '#d62728'}
for r in cross:
    x = float(r['progress']); y = float(r['mean_delta_z_mm'])
    px, py = pt(x, y, L, R, T, Bm, 0, 1, 0, 70)
    d.ellipse([px - 4, py - 4, px + 4, py + 4], fill=cols.get(r['stage'], 'black'))
for i, (s, c) in enumerate(cols.items()):
    d.rectangle([80 + i * 150, 410, 92 + i * 150, 422], fill=c)
    d.text((96 + i * 150, 408), s, font=font(10), fill='black')
img.save(FIG / 'fig03_sensitivity_by_stage.png')

# Fig4 expert
ex = list(csv.DictReader((B / 'data/expert_dynamics.csv').open(encoding='utf-8-sig')))
W, H = 1000, 450; img = Image.new('RGB', (W, H), 'white'); d = ImageDraw.Draw(img)
d.text((30, 12), 'Figure 4: expert gripper observation vs vertical velocity', font=font(16), fill='black')
L, R, T, Bm = 80, 950, 70, 370
axes(d, L, R, T, Bm, 0, 36, -400, 200, 'gripper (mm)', 'vz (mm/s)')
for i in range(0, len(ex), 2):
    g = float(ex[i]['gripper_per_finger_mm']); v = float(ex[i]['vz_mm_s'])
    px, py = pt(g, max(-400, min(200, v)), L, R, T, Bm, 0, 36, -400, 200)
    d.ellipse([px - 1, py - 1, px + 1, py + 1], fill=(31, 119, 180, 40))
# binned means
for g, v in [(14, -69), (20, 1.2), (29, 0.0), (35, 0.0)]:
    px, py = pt(g, v, L, R, T, Bm, 0, 36, -400, 200)
    d.ellipse([px - 6, py - 6, px + 6, py + 6], fill='#d62728')
    d.text((px + 8, py - 8), f'{g}mm:{v:.0f}', font=font(10), fill='#d62728')
img.save(FIG / 'fig04_expert_gripper_vz.png')

# Fig5 grip by progress
W, H = 1000, 450; img = Image.new('RGB', (W, H), 'white'); d = ImageDraw.Draw(img)
d.text((30, 12), 'Figure 5: gripper observation distribution by progress', font=font(16), fill='black')
L, R, T, Bm = 80, 950, 70, 370
axes(d, L, R, T, Bm, 0, 1, 0, 1, 'normalized progress', 'fraction of frames')
bins = [(0, .15), (.15, .3), (.3, .5), (.5, 1.01)]
for lo, hi in bins:
    sub = [r for r in ex if lo <= float(r['progress']) < hi]
    n = len(sub)
    # fraction grip >=20mm
    f = sum(float(r['gripper_per_finger_mm']) >= 20 for r in sub) / max(1, n)
    px, py = pt((lo + min(hi, 1)) / 2, f, L, R, T, Bm, 0, 1, 0, 1)
    d.rectangle([px - 40, py, px + 40, Bm], fill='#9467bd')
    d.text((px - 25, py - 16), f'{f:.2f}', font=font(10), fill='black')
    d.text((px - 25, Bm + 6), f'{lo:.2f}-{min(hi,1):.2f}', font=font(8), fill='black')
img.save(FIG / 'fig05_gripper_by_stage.png')

# Fig6 coverage
cov = list(csv.DictReader((B / 'data/coverage_summary.csv').open(encoding='utf-8-sig')))
c03 = {r['state_id']: r for r in cov if abs(float(r['radius_rad']) - 0.3) < 1e-6}
states = [r['state_id'] for r in sorted(cross, key=lambda r: float(r['progress']))]
W, H = 1100, 450; img = Image.new('RGB', (W, H), 'white'); d = ImageDraw.Draw(img)
d.text((30, 12), 'Figure 6: local training coverage at radius 0.30 rad', font=font(16), fill='black')
L, R, T, Bm = 80, 1050, 70, 340
axes(d, L, R, T, Bm, 0, len(states), 0, 1, 'states sorted by progress', 'normalized count')
maxf = max(int(c03[s]['n_frames']) for s in states); maxn = max(int(c03[s]['near35_frames']) for s in states)
for i, s in enumerate(states):
    x = L + (R - L) * i / len(states); w = (R - L) / len(states) * 0.7
    f = int(c03[s]['n_frames']) / maxf; n = int(c03[s]['near35_frames']) / max(1, maxn)
    d.rectangle([x, Bm - (Bm - T) * f, x + w, Bm], fill='#1f77b4')
    d.rectangle([x + w, Bm - (Bm - T) * n, x + 2 * w, Bm], fill='#d62728')
d.rectangle([80, 390, 92, 402], fill='#1f77b4'); d.text((96, 388), 'frames (norm)', font=font(10), fill='black')
d.rectangle([250, 390, 262, 402], fill='#d62728'); d.text((266, 388), 'near-35mm frames (norm)', font=font(10), fill='black')
img.save(FIG / 'fig06_coverage_distribution.png')

# Fig7 coverage vs sens
W, H = 1100, 450; img = Image.new('RGB', (W, H), 'white'); d = ImageDraw.Draw(img)
d.text((30, 12), 'Figure 7: sensitivity vs local coverage', font=font(16), fill='black')
for pi, (key, lab) in enumerate([('coverage_0.30_frames', 'local frames @0.30'), ('coverage_0.30_near35', 'near-35mm frames @0.30')]):
    L = 80 + pi * 520; R = L + 430; T = 70; Bm = 360
    xs = [float(r[key]) for r in cross]; ys = [float(r['mean_delta_z_mm']) for r in cross]
    xmax = max(xs) * 1.05 or 1; ymax = max(ys) * 1.05
    axes(d, L, R, T, Bm, 0, xmax, 0, ymax, lab, 'delta z step10 (mm)')
    for x, y in zip(xs, ys):
        px, py = pt(x, y, L, R, T, Bm, 0, xmax, 0, ymax)
        d.ellipse([px - 4, py - 4, px + 4, py + 4], fill='#2ca02c')
img.save(FIG / 'fig07_coverage_vs_sensitivity.png')
print('done')
