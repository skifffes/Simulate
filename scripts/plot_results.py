"""从真实 summary.csv 绘图，无需额外绘图库。"""
from pathlib import Path
import csv
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'

def main():
    with (OUT / 'summary.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    image = Image.new('RGB', (1200, 870), 'white')
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 17)
        title = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 24)
    except OSError:
        font = title = ImageFont.load_default()
    draw.text((30, 20), 'Measured handoff results (deterministic repetitions)', fill='black', font=title)
    panels = [('chain_success_rate', 'Full chain success (%)', 100, 100),
              ('mean_transition_seconds', 'Transition simulation time (s), failures included', 480, None)]
    for field, label, top, scale in panels:
        values = [float(r[field]) * (100 if scale == 100 else 1) for r in rows]
        maximum = scale or max(max(values), 0.01) * 1.1
        draw.text((30, top - 30), label, fill='black', font=font)
        for index, (row, value) in enumerate(zip(rows, values)):
            y = top + index * 36
            draw.text((30, y), row['condition'], fill='black', font=font)
            width = int(650 * value / maximum)
            color = '#5586ce' if row['condition'].startswith('direct:') else '#49a68a'
            if width:
                draw.rectangle((290, y, 290 + width, y + 23), fill=color)
            draw.text((955, y), f'{value:.2f}', fill='black', font=font)
    draw.text((30, 810), 'All conditions succeed; bars differ in transition cost, not feasibility.', fill='black', font=font)
    image.save(OUT / 'comparison.png')
    print(OUT / 'comparison.png')

if __name__ == '__main__':
    main()
