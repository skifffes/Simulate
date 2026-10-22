"""训练数据与评测渲染的一致性检查。

数据采集脚本默认 supersample=1（直接 256 渲染），而旧评测用 CameraRig 默认
supersample=2（512 渲染后 LANCZOS 降到 256）。本脚本在完全相同的场景/种子上
分别用两种方式渲染，并与数据集视频首帧做像素级对比，量化：

1. 超采样差异；
2. h264 编码噪声底（把 ss=1 渲染自己编码再解码，作为对照）。

用法：
    python scripts/check_render_consistency.py --episodes 5 --out results/experiment_1_1/render_check
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import imageio_ffmpeg  # noqa: E402

from simulation import Simulation  # noqa: E402
from vla.cameras import CameraRig  # noqa: E402
from vla.scenario import build_scene, scenario_for_seed  # noqa: E402

VIDEO = ROOT / 'data/lerobot_panda_pick_place/videos/chunk-000/observation.images.top'


def decode_first_frame(path: Path) -> np.ndarray:
    gen = imageio_ffmpeg.read_frames(str(path), pix_fmt='rgb24')
    meta = next(gen)
    frame = next(gen)
    gen.close()
    h, w = meta['size']
    return np.frombuffer(frame, dtype=np.uint8).reshape(h, w, 3).copy()


def encode_decode(img: np.ndarray, crf: int = 28) -> np.ndarray:
    tmp = ROOT / 'results/_codec_probe.mp4'
    tmp.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(tmp), img.shape[1::-1], fps=20, codec='libx264',
        output_params=['-crf', str(crf)])
    writer.send(None)
    writer.send(np.ascontiguousarray(img))
    writer.close()
    return decode_first_frame(tmp)


def metrics(a: np.ndarray, b: np.ndarray) -> dict:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    diff = a - b
    mse = float(np.mean(diff ** 2))
    gray_a = a.mean(axis=2)
    gray_b = b.mean(axis=2)
    gy, gx = np.gradient(gray_a)
    mag_a = np.hypot(gx, gy)
    gy, gx = np.gradient(gray_b)
    mag_b = np.hypot(gx, gy)
    return {
        'mae': float(np.mean(np.abs(diff))),
        'mse': mse,
        'psnr': float(10 * np.log10(255.0 ** 2 / max(mse, 1e-9))),
        'grad_mean_ratio': float(mag_a.mean() / max(mag_b.mean(), 1e-9)),
        'strong_grad_frac': float(np.mean(mag_a > 40.0)),
        'strong_grad_frac_ref': float(np.mean(mag_b > 40.0)),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=5)
    ap.add_argument('--seed-base', type=int, default=20261022,
                    help='episode 0 的采集种子（collect 的 seed+attempts）')
    ap.add_argument('--warmup-steps', type=int, default=30)
    ap.add_argument('--pose-match', action='store_true', default=True,
                    help='用数据集记录的本体状态复原首帧姿态')
    ap.add_argument('--no-pose-match', dest='pose_match', action='store_false')
    ap.add_argument('--out', default='results/experiment_1_1/render_check')
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, comps = [], []
    for i in range(args.episodes):
        video = VIDEO / f'episode_{i:06d}.mp4'
        if not video.exists():
            print('skip missing', video)
            continue
        data = decode_first_frame(video)
        seed = args.seed_base + i
        scenario = scenario_for_seed(seed)
        sim, block = build_scene(scenario)
        try:
            if args.pose_match:
                # 用数据集记录的本体状态精确复原首帧姿态，排除“时刻不同”的干扰
                import pyarrow.parquet as pq
                pq_path = ROOT / f'data/lerobot_panda_pick_place/data/chunk-000/episode_{i:06d}.parquet'
                table = pq.read_table(pq_path)
                q0 = table['observation.state'][0].as_py()
                import pybullet as p
                for j, v in enumerate(q0[:7]):
                    p.resetJointState(sim.robot, j + 1, float(v), physicsClientId=sim.client)
                for k, link in enumerate((9, 10)):
                    p.resetJointState(sim.robot, link, float(q0[7 + k]),
                                      physicsClientId=sim.client)
            else:
                for _ in range(args.warmup_steps):
                    sim.step()
            shots = {}
            for ss in (1, 2):
                cam = CameraRig(sim, supersample=ss)
                shots[ss] = np.asarray(cam.capture()['top'], dtype=np.uint8).copy()
            recoded = encode_decode(shots[1])
        finally:
            sim.close()

        pairs = {
            'data_vs_ss1': metrics(data, shots[1]),
            'data_vs_ss2': metrics(data, shots[2]),
            'data_vs_ss1_recoded': metrics(data, recoded),
            'ss1_vs_codec_floor': metrics(shots[1], recoded),
        }
        rows.append({'episode': i, 'seed': seed,
                     **{f'{k}_{m}': v for k, d in pairs.items() for m, v in d.items()}})

        gap = np.abs(data.astype(int) - shots[1].astype(int)).sum(axis=2)
        gap2 = np.abs(data.astype(int) - shots[2].astype(int)).sum(axis=2)
        tile = np.concatenate([data, shots[1], shots[2],
                               np.clip(gap, 0, 255).astype(np.uint8)[..., None].repeat(3, 2),
                               np.clip(gap2, 0, 255).astype(np.uint8)[..., None].repeat(3, 2)],
                              axis=1)
        comps.append(tile)
        print(f'ep{i}: mae data~ss1={pairs["data_vs_ss1"]["mae"]:.2f} '
              f'data~ss2={pairs["data_vs_ss2"]["mae"]:.2f} '
              f'grad_ratio ss1={pairs["data_vs_ss1"]["grad_mean_ratio"]:.3f}')

    if comps:
        Image.fromarray(np.concatenate(comps, axis=0)).save(out / 'render_compare.png')
    with (out / 'metrics.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out / 'summary.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')

    if rows:
        def avg(key):
            return float(np.mean([r[key] for r in rows]))
        summary = {
            'n_episodes': len(rows),
            'data_vs_ss1_mae': avg('data_vs_ss1_mae'),
            'data_vs_ss2_mae': avg('data_vs_ss2_mae'),
            'data_vs_ss1_psnr': avg('data_vs_ss1_psnr'),
            'data_vs_ss2_psnr': avg('data_vs_ss2_psnr'),
            'codec_floor_mae': avg('ss1_vs_codec_floor_mae'),
            'grad_ratio_ss1': avg('data_vs_ss1_grad_mean_ratio'),
            'grad_ratio_ss2': avg('data_vs_ss2_grad_mean_ratio'),
            'conclusion': ('采集与 ss=1 一致' if avg('data_vs_ss1_mae') < avg('data_vs_ss2_mae')
                           else '采集与 ss=2 一致'),
        }
        (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
