"""从 LeRobot 训练日志里抽出 loss 曲线，输出 CSV + 自包含 SVG（只依赖标准库）。

用法：
    python server/plot_training_curve.py --log results/experiment_1/training/train_smolvla.log \
        --out results/experiment_1/training
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import os
import re

# step:15K smpl:119K ... loss:0.009 grdn:0.247 lr:5.2e-05
LINE_RE = re.compile(
    r"step:([0-9.]+[KkMm]?)\b.*?loss:([0-9.eE+-]+).*?grdn:([0-9.eE+-]+).*?lr:([0-9.eE+-]+)"
)


def _to_int(text: str) -> int:
    if text[-1] in "Kk":
        return int(float(text[:-1]) * 1_000)
    if text[-1] in "Mm":
        return int(float(text[:-1]) * 1_000_000)
    return int(float(text))


def parse_log(path: str, log_freq: int = 50) -> list[dict]:
    """日志里的 step 只保留两位有效数字（1K / 15K），因此用行序号按 log_freq 还原。"""
    parsed: list[dict] = []
    with io.open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            match = LINE_RE.search(line)
            if not match:
                continue
            parsed.append(
                {
                    "step_logged": _to_int(match.group(1)),
                    "loss": float(match.group(2)),
                    "grad_norm": float(match.group(3)),
                    "lr": float(match.group(4)),
                }
            )
    if not parsed:
        raise SystemExit(f"没有在 {path} 里找到 loss 行")

    rows: list[dict] = []
    for i, row in enumerate(parsed):
        step = log_freq * (i + 1)
        rows.append({"step": step, **row})
        # 日志 step 在 >=1000 后按 "%.0fK" 打印，量化单位是 1000 步
        tol = 500.0 if step >= 1000 else 1.0
        if abs(row["step_logged"] - step) > tol:
            print(f"警告：第 {i + 1} 行日志 step={row['step_logged']} 与推算 {step} 不一致")
    return rows


def smooth(values: list[float], window: int = 5) -> list[float]:
    out: list[float] = []
    for i in range(len(values)):
        lo = max(0, i - window // 2)
        hi = min(len(values), lo + window)
        chunk = values[lo:hi]
        out.append(sum(chunk) / len(chunk))
    return out


def write_csv(rows: list[dict], path: str) -> None:
    with io.open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["step", "step_logged", "loss", "loss_ma5", "grad_norm", "lr"],
            extrasaction="ignore",
        )
        writer.writeheader()
        ma = smooth([r["loss"] for r in rows])
        for row, m in zip(rows, ma):
            writer.writerow({**row, "loss_ma5": f"{m:.6f}"})


def write_svg(rows: list[dict], path: str, title: str) -> None:
    width, height = 900, 500
    left, right, top, bottom = 90, 30, 60, 70
    plot_w = width - left - right
    plot_h = height - top - bottom

    xs = [r["step"] for r in rows]
    losses = [r["loss"] for r in rows]
    ma = smooth(losses)

    x_min, x_max = 0.0, float(max(xs))
    y_lo = min(losses) * 0.85
    y_hi = max(losses) * 1.15

    def sx(x: float) -> float:
        return left + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y: float) -> float:
        y = max(y, y_lo)
        frac = (math.log10(y) - math.log10(y_lo)) / (math.log10(y_hi) - math.log10(y_lo))
        return top + (1.0 - frac) * plot_h

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="DejaVu Sans, Arial, sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{left}" y="32" font-size="18" fill="#111827">{title}</text>',
    ]

    # y 网格（1/2/5 × 10^k 刻度）
    ticks = sorted(
        mantissa * 10 ** exp
        for exp in range(-4, 1)
        for mantissa in (1, 2, 5)
        if y_lo <= mantissa * 10 ** exp <= y_hi
    )
    for tick in ticks:
        y = sy(tick)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" '
            f'stroke="#e5e7eb" stroke-width="1"/>'
        )
        label = f"{tick:g}"
        parts.append(
            f'<text x="{left - 12}" y="{y + 4:.1f}" font-size="12" fill="#4b5563" '
            f'text-anchor="end">{label}</text>'
        )

    # x 网格
    n_ticks = 6
    for i in range(n_ticks + 1):
        x_val = x_max * i / n_ticks
        x = sx(x_val)
        parts.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" '
            f'stroke="#f3f4f6" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{top + plot_h + 22}" font-size="12" fill="#4b5563" '
            f'text-anchor="middle">{int(x_val)}</text>'
        )

    parts.append(
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" '
        f'stroke="#9ca3af" stroke-width="1"/>'
    )

    raw_pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, losses))
    ma_pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ma))
    parts.append(f'<polyline points="{raw_pts}" fill="none" stroke="#bfdbfe" stroke-width="1.5"/>')
    parts.append(f'<polyline points="{ma_pts}" fill="none" stroke="#1d4ed8" stroke-width="2.2"/>')

    parts.append(
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 18}" font-size="13" fill="#374151" '
        f'text-anchor="middle">training step</text>'
    )
    parts.append(
        f'<text x="24" y="{top + plot_h / 2:.1f}" font-size="13" fill="#374151" '
        f'text-anchor="middle" transform="rotate(-90 24 {top + plot_h / 2:.1f})">loss (log scale)</text>'
    )

    # 图例
    lx, ly = left + plot_w - 200, top + 18
    parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 26}" y2="{ly}" stroke="#bfdbfe" stroke-width="2"/>')
    parts.append(f'<text x="{lx + 34}" y="{ly + 4}" font-size="12" fill="#374151">raw (log_freq=50)</text>')
    parts.append(f'<line x1="{lx}" y1="{ly + 20}" x2="{lx + 26}" y2="{ly + 20}" stroke="#1d4ed8" stroke-width="2.2"/>')
    parts.append(f'<text x="{lx + 34}" y="{ly + 24}" font-size="12" fill="#374151">moving average (5)</text>')

    first, last = rows[0], rows[-1]
    parts.append(
        f'<text x="{left}" y="{top - 12}" font-size="12" fill="#6b7280">'
        f"step {first['step']}: loss {first['loss']:.3f} → step {last['step']}: loss {last['loss']:.3f}</text>"
    )

    parts.append("</svg>")
    with io.open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(parts) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", default="results/experiment_1/training/train_smolvla.log")
    parser.add_argument("--out", default="results/experiment_1/training")
    parser.add_argument("--title", default="SmolVLA fine-tuning loss (15k steps, batch 8)")
    parser.add_argument("--log-freq", type=int, default=50, help="日志打印间隔（步）")
    args = parser.parse_args()

    rows = parse_log(args.log, args.log_freq)
    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "training_curve.csv")
    svg_path = os.path.join(args.out, "training_loss.svg")
    write_csv(rows, csv_path)
    write_svg(rows, svg_path, args.title)

    losses = [r["loss"] for r in rows]
    print(f"记录点 {len(rows)} 个，step {rows[0]['step']}–{rows[-1]['step']}")
    print(f"loss 首次 {losses[0]:.4f} / 最低 {min(losses):.4f} / 末次 {losses[-1]:.4f}")
    print(f"已写出 {csv_path} 和 {svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
