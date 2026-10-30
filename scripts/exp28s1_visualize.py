import csv, math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(r'D:\python\simulate')
B = ROOT / 'results/experiment_2_8/supplement_s1'
D = B / 'data'; F = B / 'figures'; F.mkdir(parents=True, exist_ok=True)
FONT = 'C:/Windows/Fonts/arial.ttf'


def font(s):
    try: return ImageFont.truetype(FONT, s)
    except Exception: return ImageFont.load_default()


def frame(title, w=1100, h=470):
    img = Image.new('RGB', (w, h), 'white'); d = ImageDraw.Draw(img)
    d.text((30, 12), title, font=font(16), fill='black'); return img, d


def axes(d, L, R, T, Bm, xmin, xmax, ymin, ymax, xlab, ylab):
    d.rectangle([L, T, R, Bm], outline='black')
    for i in range(5):
        y = Bm - (Bm - T) * i / 4
        d.line([L, y, R, y], fill=(225, 225, 225))
        d.text((8, y - 6), f'{ymin + (ymax - ymin) * i / 4:.4g}', font=font(8), fill='black')
    for i in range(5):
        x = L + (R - L) * i / 4
        d.text((x - 10, Bm + 6), f'{xmin + (xmax - xmin) * i / 4:.3g}', font=font(8), fill='black')
    d.text((L, Bm + 20), xlab, font=font(9), fill='black')
    d.text((L, T - 12), ylab, font=font(9), fill='black')


def pt(x, y, L, R, T, Bm, xmin, xmax, ymin, ymax):
    return (L + (R - L) * (x - xmin) / (xmax - xmin),
            Bm - (Bm - T) * (y - ymin) / (ymax - ymin))


dyn = list(csv.DictReader((D / 'expert_target_dynamics.csv').open(encoding='utf-8-sig')))
chunk = list(csv.DictReader((D / 'expert_chunk_targets.csv').open(encoding='utf-8-sig')))
ev = list(csv.DictReader((D / 'episode_timing_events.csv').open(encoding='utf-8-sig')))
timing = list(csv.DictReader((D / 'timing_summary.csv').open(encoding='utf-8-sig')))
cond = list(csv.DictReader((D / 'conditional_target_statistics.csv').open(encoding='utf-8-sig')))
comp = list(csv.DictReader((D / 'model_expert_comparison.csv').open(encoding='utf-8-sig')))
model = list(csv.DictReader((ROOT / 'results/experiment_2_8/data/cross_state_analysis.csv').open(encoding='utf-8-sig')))
G = np.array([float(r['gripper_obs_mm']) for r in dyn])
Z = np.array([float(r['z_target_mm']) for r in dyn])
DZ = np.array([float(r['dz_target_mm']) for r in dyn])
V = np.array([float(r['vz_actual_mm_s']) for r in dyn])
P = np.array([float(r['progress']) for r in dyn])

# Fig1 two panels
img, d = frame('Figure 1: gripper vs expert target height', 1200)
for pi, (vals, lab, ymax, ymin) in enumerate([(Z, 'target EE z (mm)', 600, 0), (DZ, 'target dz vs current (mm)', 30, -80)]):
    L = 60 + pi * 590; R = L + 520; T = 70; Bm = 370
    axes(d, L, R, T, Bm, 0, 38, ymin, ymax, 'gripper (mm)', lab)
    for i in range(0, len(G), 2):
        x, y = pt(G[i], max(ymin, min(ymax, vals[i])), L, R, T, Bm, 0, 38, ymin, ymax)
        d.ellipse([x - 1, y - 1, x + 1, y + 1], fill=(31, 119, 180))
    for g, y0 in [(14, np.mean(vals[(G >= 13) & (G < 15.5)])), (20, np.mean(vals[(G >= 15.5) & (G < 25)])),
                  (29, np.mean(vals[(G >= 25) & (G < 35)])), (35, np.mean(vals[(G >= 35) & (G < 41)]))]:
        x, y = pt(g, max(ymin, min(ymax, y0)), L, R, T, Bm, 0, 38, ymin, ymax)
        d.ellipse([x - 5, y - 5, x + 5, y + 5], fill='#d62728')
img.save(F / 'fig01_gripper_target_distribution.png')

# Fig2 target vs actual
img, d = frame('Figure 2: expert target dz vs actual vertical velocity')
remap = np.argsort(np.argsort(P))
L, R, T, Bm = 80, 1000, 70, 370
axes(d, L, R, T, Bm, -100, 40, -350, 150, 'expert target dz (mm)', 'actual vz (mm/s)')
for i in range(0, len(DZ), 2):
    x, y = pt(max(-100, min(40, DZ[i])), max(-350, min(150, V[i])), L, R, T, Bm, -100, 40, -350, 150)
    d.ellipse([x - 1, y - 1, x + 1, y + 1], fill=(31, 119, 180))
img.save(F / 'fig02_target_vs_actual_motion.png')

# Fig3 timelines three episodes
sel = [60, 66, 96]
by_ep = {e: [r for r in dyn if int(r['episode_id']) == e] for e in sel}
img, d = frame('Figure 3: episode timelines (grip, actual/target height, target dz)', 1200, 760)
for ei, e in enumerate(sel):
    rs = by_ep[e]
    t = np.array([float(r['timestamp']) for r in rs])
    g = np.array([float(r['gripper_obs_mm']) for r in rs])
    zo = np.array([float(r['z_obs_mm']) for r in rs])
    zt = np.array([float(r['z_target_mm']) for r in rs])
    dz = np.array([float(r['dz_target_mm']) for r in rs])
    T = 60 + ei * 230; Bm = T + 160
    for ci, (vals, lab, lo, hi, col) in enumerate([
            (g, 'grip (mm)', 0, 38, '#2ca02c'),
            (zo, 'EE z (mm)', 0, 520, '#1f77b4'),
            (dz, 'target dz (mm)', -80, 20, '#d62728')]):
        L = 60 + ci * 390; R = L + 350
        axes(d, L, R, T, Bm, 0, 6.5, lo, hi, 'time (s)', lab)
        d.line([pt(x, max(lo, min(hi, y)), L, R, T, Bm, 0, 6.5, lo, hi) for x, y in zip(t, vals)], fill=col, width=2)
        if ci == 1:
            d.line([pt(x, max(lo, min(hi, y)), L, R, T, Bm, 0, 6.5, lo, hi) for x, y in zip(t, zt)], fill='#ff7f0e', width=1)
    d.text((460, T + 2), f'ep{e}', font=font(11), fill='black')
img.save(F / 'fig03_episode_timeline.png')

# Fig4 timing hist
img, d = frame('Figure 4: gripper close -> descent delay (mid threshold)')
vals = [float(r['close_to_target_mid_s']) for r in ev if r['close_to_target_mid_s'] != '']
a = [float(r['close_to_actual_mid_s']) for r in ev if r['close_to_actual_mid_s'] != '']
L, R, T, Bm = 80, 1000, 70, 370
axes(d, L, R, T, Bm, 0, 2.0, 0, 30, 'delay (s)', 'count')
for vals2, col, off in ((vals, '#d62728', -0.02), (a, '#1f77b4', 0.02)):
    hist, edges = np.histogram(vals2, bins=np.arange(0, 2.1, 0.1))
    for i, h in enumerate(hist):
        x0, y0 = pt(edges[i] + off, 0, L, R, T, Bm, 0, 2, 0, 30)
        x1, y1 = pt(edges[i + 1] + off, h, L, R, T, Bm, 0, 2, 0, 30)
        d.rectangle([x0, y1, x1, y0], fill=col)
img.save(F / 'fig04_event_timing_distribution.png')

# Fig5 target dz h9 vs progress
valid = [r for r in chunk if int(r['horizon']) == 9 and int(r['valid']) == 1]
img, d = frame('Figure 5: expert future target dz (h=9) vs progress')
L, R, T, Bm = 80, 1000, 70, 370
axes(d, L, R, T, Bm, 0, 1, -300, 50, 'progress', 'target dz h9 (mm)')
# progress not in chunk; derive by episode frame max
for r in dyn:
    r['_p'] = float(r['progress'])
pmap = {(int(r['episode_id']), int(r['frame_id'])): float(r['progress']) for r in dyn}
for i in range(0, len(valid), 2):
    x, y = pt(pmap.get((int(valid[i]['episode_id']), int(valid[i]['frame_id'])), 0), max(-300, min(50, float(valid[i]['dz_target_mm']))), L, R, T, Bm, 0, 1, -300, 50)
    d.ellipse([x - 1, y - 1, x + 1, y + 1], fill=(31, 119, 180))
img.save(F / 'fig05_target_by_progress.png')

# Fig6 conditional: early progress only
img, d = frame('Figure 6: conditional target dz (common support only)', 1100)
L, R, T, Bm = 80, 1000, 70, 370
axes(d, L, R, T, Bm, 0, 4, -20, 5, '', 'target dz h0 (mm)')
early = [r for r in cond if r['progress_bin'] == '0.00-0.15']
labels = ['14', '16-25', '25-35', '35-40']
for i, r in enumerate(early):
    x = i + 0.5; y = float(r['dz_target_mean_mm'])
    px, py = pt(x, y, L, R, T, Bm, 0, 4, -20, 5)
    d.rectangle([px - 35, py, px + 35, Bm], fill='#9467bd')
    d.text((px - 20, py - 14), f'{y:.1f}', font=font(9), fill='black')
    d.text((px - 20, Bm + 8), labels[i], font=font(9), fill='black')
d.text((100, 410), 'later progress: only 14mm (and 19 frames of 16-25) -> no 14->40 common support', font=font(9), fill='black')
img.save(F / 'fig06_conditional_target_analysis.png')

# Fig7 model vs expert
m = np.array([float(r['model_delta_z_mm']) for r in comp]); t = np.array([float(r['local_target_dz_mm']) for r in comp]); ee = np.array([float(r['ee_z_m']) for r in comp])
img, d = frame('Figure 7: model 14->40 sensitivity vs local expert target dz')
L, R, T, Bm = 80, 1000, 70, 370
axes(d, L, R, T, Bm, -12, 5, 0, 75, 'local expert target dz (mm)', 'model delta z step10 (mm)')
for x, y, e in zip(t, m, ee):
    px, py = pt(x, y, L, R, T, Bm, -12, 5, 0, 75)
    c = int(255 * min(1, max(0, (e - 0.03) / 0.15)))
    d.ellipse([px - 4, py - 4, px + 4, py + 4], fill=(255 - c, 80, c))
d.text((650, 80), f'Pearson r = {np.corrcoef(t, m)[0,1]:.2f}', font=font(11), fill='black')
d.text((650, 100), 'color: initial EE z (blue low, red high)', font=font(9), fill='black')
img.save(F / 'fig07_model_vs_expert.png')

# Fig8 counterexamples: delays by episode + coverage/sensitivity outlier
img, d = frame('Figure 8: counterexamples and inconclusive cases', 1200, 500)
L, R, T, Bm = 60, 560, 70, 380
axes(d, L, R, T, Bm, 0, 60, 0, 2.0, 'episode (sorted)', 'close->target delay (s)')
ev2 = sorted([(float(r['close_to_target_mid_s']), int(r['episode_id'])) for r in ev if r['close_to_target_mid_s'] != ''])
for i, (v, e) in enumerate(ev2):
    x = L + (R - L) * i / max(1, len(ev2) - 1); y = Bm - (Bm - T) * v / 2.0
    d.rectangle([x - 2, y, x + 2, Bm], fill='#d62728')
    if v > 1.0:
        d.text((x - 12, y - 14), f'ep{e}', font=font(9), fill='black')
L, R, T, Bm = 650, 1150, 70, 380
axes(d, L, R, T, Bm, -12, 5, 0, 75, 'local expert target dz', 'model delta z')
for x, y in zip(t, m):
    px, py = pt(x, y, L, R, T, Bm, -12, 5, 0, 75)
    d.ellipse([px - 3, py - 3, px + 3, py + 3], fill='#2ca02c')
img.save(F / 'fig08_counterexamples.png')
print('done')
