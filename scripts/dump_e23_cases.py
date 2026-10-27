"""实验 2.3：导出代表性成功/失败案例的轨迹视频与关键帧。

案例：
    b_ok      D1 标准 B 成功（seed 40000004，yaw≈90°）
    a0b_fail  D2 A0→B 连续交接失败（seed 40000000，B 阶段抓起/搬运）

用法（服务器）：
    HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=4 PYTHONPATH=src venv/bin/python \
        scripts/dump_e23_cases.py \
        --policy outputs/smolvla_e23_D1/checkpoints/020000/pretrained_model \
        --case b_ok --seed 40000004 --out results/experiment_2_3/cases/b_ok
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pybullet as p

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from eval_handoff_timing import a_success, finger_contacts, obj_state  # noqa: E402
from eval_multi_atomic_vla import PolicyRunner, sample_scene  # noqa: E402
from manipulation import finger_widths, set_gripper  # noqa: E402
from vla.atomic_tasks.config2 import CAMERAS, IMAGE_SIZE, STEPS_PER_ACTION, TASKS  # noqa: E402
from vla.atomic_tasks.scene2 import build_scene  # noqa: E402
from vla.cameras import CameraRig  # noqa: E402
from vla.lerobot_writer import encode_video  # noqa: E402


def _save(frames, trace, out: Path, switch_tick=None):
    from PIL import Image
    out.mkdir(parents=True, exist_ok=True)
    for cam in ('top', 'wrist'):
        arr = np.stack(frames[cam])
        encode_video(arr, out / f'{cam}.mp4', 20)
        picks = {'start': 0, 'mid': len(arr) // 2, 'end': len(arr) - 1}
        if switch_tick is not None:
            picks['switch'] = min(switch_tick, len(arr) - 1)
        for tag, idx in picks.items():
            Image.fromarray(arr[idx]).save(out / f'{cam}_{tag}.png')
    (out / 'trace.json').write_text(json.dumps(trace, ensure_ascii=False), encoding='utf-8')


def run_case(runner, case: str, seed: int, out: Path, max_ticks: int):
    frames = {'top': [], 'wrist': []}
    trace = {'case': case, 'seed': seed, 'ticks': []}
    switch_tick = None

    if case == 'b_ok':
        scene = sample_scene('B', seed)
        sim, block, wall, tray, info = build_scene(seed=seed, **scene)
        camera = CameraRig(sim, CAMERAS, IMAGE_SIZE, supersample=1)
        for _ in range(30):
            sim.step()
        runner.reset()
        np.random.seed(seed)
        runner.torch.manual_seed(seed)
        runner.torch.cuda.manual_seed_all(seed)
        skill = 'B'
        for tick in range(max_ticks):
            shot = camera.capture()
            state = np.concatenate([sim.q(), np.asarray(finger_widths(sim))])
            action = runner.act(shot, state, TASKS[skill])
            target = np.clip(action[:7], np.asarray(sim.lower) + 1e-4,
                             np.asarray(sim.upper) - 1e-4)
            width = float(np.clip(action[7], 0.0, 0.04))
            sim.command(target)
            set_gripper(sim, width, 20.0, 0.05)
            for _ in range(STEPS_PER_ACTION):
                sim.step()
            for cam in frames:
                frames[cam].append(shot[cam])
            st = obj_state(sim, block)
            trace['ticks'].append({'t': tick, 'z': float(st['position'][2]),
                                   'x': float(st['position'][0]),
                                   'width': width,
                                   'contacts': len(finger_contacts(sim, block))})
        _save(frames, trace, out)
        p.disconnect(sim.client)
        return

    # A0→B 连续失败案例
    sim, block, wall, tray, info = build_scene(seed=seed, jitter=0.008, yaw=0.0)
    camera = CameraRig(sim, CAMERAS, IMAGE_SIZE, supersample=1)
    for _ in range(30):
        sim.step()
    pos0, _ = p.getBasePositionAndOrientation(block, physicsClientId=sim.client)
    initial_z = float(pos0[2])
    max_z = initial_z
    runner.reset()
    np.random.seed(seed)
    runner.torch.manual_seed(seed)
    runner.torch.cuda.manual_seed_all(seed)

    for tick in range(max_ticks):
        shot = camera.capture()
        state = np.concatenate([sim.q(), np.asarray(finger_widths(sim))])
        action = runner.act(shot, state, TASKS['A0'])
        target = np.clip(action[:7], np.asarray(sim.lower) + 1e-4,
                         np.asarray(sim.upper) - 1e-4)
        width = float(np.clip(action[7], 0.0, 0.04))
        sim.command(target)
        set_gripper(sim, width, 20.0, 0.05)
        for _ in range(STEPS_PER_ACTION):
            sim.step()
        pos, _ = p.getBasePositionAndOrientation(block, physicsClientId=sim.client)
        max_z = max(max_z, float(pos[2]))
        if tick > 20 and a_success('A0', sim, block, initial_z, max_z):
            switch_tick = tick + 1
            break
    st = obj_state(sim, block)
    trace['switch'] = {'tick': switch_tick, 'z': float(st['position'][2]),
                       'x': float(st['position'][0]), 'y': float(st['position'][1])}
    runner.reset()
    for tick in range(max_ticks):
        shot = camera.capture()
        state = np.concatenate([sim.q(), np.asarray(finger_widths(sim))])
        action = runner.act(shot, state, TASKS['B'])
        target = np.clip(action[:7], np.asarray(sim.lower) + 1e-4,
                         np.asarray(sim.upper) - 1e-4)
        width = float(np.clip(action[7], 0.0, 0.04))
        sim.command(target)
        set_gripper(sim, width, 20.0, 0.05)
        for _ in range(STEPS_PER_ACTION):
            sim.step()
        for cam in frames:
            frames[cam].append(shot[cam])
        st = obj_state(sim, block)
        trace['ticks'].append({'t': tick, 'z': float(st['position'][2]),
                               'x': float(st['position'][0]),
                               'width': width,
                               'contacts': len(finger_contacts(sim, block))})
    _save(frames, trace, out, switch_tick=0)
    p.disconnect(sim.client)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', required=True)
    ap.add_argument('--case', choices=['b_ok', 'a0b_fail'], required=True)
    ap.add_argument('--seed', type=int, default=40000000)
    ap.add_argument('--max-ticks', type=int, default=260)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    runner = PolicyRunner(args.policy)
    run_case(runner, args.case, args.seed, args.out, args.max_ticks)
    print('saved', args.out)


if __name__ == '__main__':
    main()
