"""用训练好的 SmolVLA 策略在 PyBullet 抓放场景里做闭环评测。

策略每 1/20 s 观测一次 top/wrist 图像 + 9 维状态，输出 8 维动作
(7 个绝对关节目标 + 夹爪宽度)，与采集脚本 collect_atomic_vla_data.py 的执行方式完全一致。

用法示例::

    python scripts/eval_vla_policy.py \
        --policy outputs/smolvla_panda/checkpoints/010000/pretrained_model \
        --episodes 20 --out results/experiment_1/eval_smolvla_10k

训练在服务器上完成时，直接把整个仓库 + checkpoint 拷到服务器无头运行即可。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from grasp_task import DEFAULT_PARAMS, FAIL_TEXT, GraspPlaceTask  # noqa: E402
from manipulation import (  # noqa: E402
    FINGER_MAX_WIDTH,
    finger_widths,
    object_state,
    relative_pose,
    set_gripper,
)
from vla.cameras import CameraRig  # noqa: E402
from vla.config import CAMERAS, FPS, IMAGE_SIZE, STEPS_PER_ACTION, TASK  # noqa: E402
from vla.scenario import build_scene, scenario_for_seed  # noqa: E402

ACTION_DIM = 8
STATE_DIM = 9


# --------------------------------------------------------------------------
# 策略封装
# --------------------------------------------------------------------------
class SmolVLARunner:
    """把 lerobot 的 SmolVLAPolicy 包成 act(frames, state) -> np.ndarray。"""

    def __init__(self, path: str, device: str = 'cuda', image_keys=None, n_action_steps=None):
        import torch
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        self.torch = torch
        self.device = torch.device(device)
        # 微调后的 checkpoint 里同时保存了 Normalize/Unnormalize 的统计量，
        # 直接 from_pretrained 即可；基座权重没有统计量（mean/std 为 inf），不能用于评测。
        self.policy = SmolVLAPolicy.from_pretrained(str(path))
        if n_action_steps:
            # 训练时按 50 步 chunk 优化，但闭环执行时可以只走前 n 步就重新规划，
            # 用更频繁的反馈抑制开环漂移（chunk_size 保持不变）。
            self.policy.config.n_action_steps = int(n_action_steps)
        self.policy.to(self.device)
        self.policy.eval()
        self.image_keys = list(image_keys or CAMERAS.keys())
        self.calls = 0

    def reset(self):
        self.policy.reset()

    def act(self, frames, state):
        torch = self.torch
        batch = {}
        for name in self.image_keys:
            img = frames[name]
            x = torch.from_numpy(np.ascontiguousarray(img)).to(self.device)
            x = x.float().div_(255.0).permute(2, 0, 1).unsqueeze(0)
            batch[f'observation.images.{name}'] = x
        batch['observation.state'] = (
            torch.from_numpy(np.asarray(state, dtype=np.float32)).to(self.device).unsqueeze(0)
        )
        batch['task'] = [TASK]
        with torch.no_grad():
            action = self.policy.select_action(batch)
        self.calls += 1
        action = action.squeeze(0).float().cpu().numpy().reshape(-1)
        if action.shape[0] < ACTION_DIM:
            pad = np.zeros(ACTION_DIM - action.shape[0], dtype=np.float32)
            action = np.concatenate([action, pad])
        return action[:ACTION_DIM]


class ExpertReplayRunner:
    """调试用：直接读数据集里的动作，代替策略验证评测管线本身是否正常。"""

    def __init__(self, dataset_root: str, episode: int):
        import json

        import pyarrow.parquet as pq

        root = Path(dataset_root)
        info = json.loads((root / 'meta/info.json').read_text(encoding='utf-8'))
        path = root / info['data_path'].format(
            episode_chunk=int(episode) // info['chunks_size'], episode_index=int(episode))
        if not path.exists():
            raise FileNotFoundError(f'找不到 parquet: {path}')
        data = pq.read_table(path).to_pydict()
        rows = [i for i, ep in enumerate(data['episode_index']) if int(ep) == int(episode)]
        if not rows:
            raise ValueError(f'episode {episode} 不在 {path.name} 中')
        self.actions = np.asarray([data['action'][i] for i in rows], dtype=np.float32)
        self.calls = 0

    def reset(self):
        self.calls = 0

    def act(self, frames, state):
        i = min(self.calls, len(self.actions) - 1)
        self.calls += 1
        return self.actions[i]


# --------------------------------------------------------------------------
# 单集 rollout
# --------------------------------------------------------------------------
def _write_video(path: Path, frames, fps: int = 20):
    """把俯视相机画面编码成 H.264，便于人眼复核成功/失败过程。"""
    import av

    container = av.open(str(path), mode='w')
    try:
        stream = container.add_stream('libx264', rate=fps)
        stream.width, stream.height = frames[0].shape[1], frames[0].shape[0]
        stream.pix_fmt = 'yuv420p'
        stream.options = {'crf': '28'}
        for image in frames:
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(image, dtype=np.uint8),
                                               format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def run_episode(runner, seed: int, args) -> dict:
    scenario = scenario_for_seed(seed, args.jitter, args.friction_jitter)
    sim, block = build_scene(scenario)
    try:
        camera = CameraRig(sim)
        for _ in range(args.warmup_steps):
            sim.step()

        task = GraspPlaceTask(sim, None, block, params=scenario.params,
                              record_interval=4, contact_interval=4)
        runner.reset()
        initial = task.initial
        max_z = float(initial['position'][2])
        grasp_local = None
        slip = 0.0
        actions, states, tick_err, act_latency = [], [], [], []
        video = [] if args.record_video else None
        closed_once = False

        start = time.perf_counter()
        for tick in range(args.max_ticks):
            frames = camera.capture()
            if video is not None:
                video.append(np.asarray(frames['top'], dtype=np.uint8).copy())
            state = np.concatenate([sim.q(), np.asarray(finger_widths(sim))])
            tick_start = time.perf_counter()
            action = runner.act(frames, state)
            act_latency.append(time.perf_counter() - tick_start)
            if not np.all(np.isfinite(action)):
                action = np.nan_to_num(action)

            target = np.clip(action[:7], np.asarray(sim.lower) + 1e-4,
                             np.asarray(sim.upper) - 1e-4)
            width = float(np.clip(action[7], 0.0, FINGER_MAX_WIDTH))

            actions.append(action.tolist())
            states.append(state.tolist())
            tick_err.append(float(np.max(np.abs(np.asarray(sim.q()) - target))))

            sim.command(target)
            set_gripper(sim, width, DEFAULT_PARAMS['grasp_force'], DEFAULT_PARAMS['close_speed'])
            for _ in range(STEPS_PER_ACTION):
                sim.step()

            obj = object_state(sim, block)
            max_z = max(max_z, float(obj['position'][2]))

            # 工件被抬离桌面 2 cm 后，把它相对夹爪的位姿作为基准，
            # 之后的最大偏移即运输过程中的滑移量（与专家判定的 slip 同义）。
            if not closed_once and float(obj['position'][2]) - float(initial['position'][2]) > 0.02:
                closed_once = True
                grasp_local = relative_pose(sim, block)
            if grasp_local is not None:
                local = relative_pose(sim, block)
                slip = max(slip, float(np.linalg.norm(np.asarray(local[0]) - np.asarray(grasp_local[0]))))
            task._sample('policy')

            # 放下并保持：工件高度回到初始位置附近且夹爪已张开，则提前结束
            if tick > 40 and obj['position'][2] < initial['position'][2] + 0.01 and width > 0.035:
                if float(obj['linear_speed']) < DEFAULT_PARAMS['rest_linear_speed']:
                    break
        rollout_seconds = time.perf_counter() - start

        # 收尾：张开夹爪、原地静置，让工件稳定下来
        set_gripper(sim, DEFAULT_PARAMS['open_width'],
                    DEFAULT_PARAMS['open_force'], DEFAULT_PARAMS['open_speed'])
        settle_start = sim.steps
        settle_interval = max(1, int(round(0.05 / sim.dt)))
        for _ in range(int(round(DEFAULT_PARAMS['settle_seconds'] / sim.dt))):
            sim.step()
            if (sim.steps - settle_start) % settle_interval == 0:
                task._sample('settle')
                if video is not None:
                    video.append(np.asarray(camera.capture()['top'], dtype=np.uint8).copy())
        for s in task.samples:
            if s['t'] >= (settle_start - task.step0) * sim.dt:
                task.max_rest_speed = max(task.max_rest_speed, s['obj_lin_speed'])

        task.lift_height = max_z - float(initial['position'][2])
        task.slip = slip
        task._finalize()

        result = task.result()
        result.update({
            'seed': int(seed),
            'ticks': int(tick + 1),
            'wall_seconds': round(rollout_seconds, 3),
            'mean_tracking_error_rad': float(np.mean(tick_err)),
            'max_tracking_error_rad': float(np.max(tick_err)),
            'inference_latency_ms_mean': float(np.mean(act_latency) * 1000),
            'inference_latency_ms_p95': float(np.percentile(act_latency, 95) * 1000),
            'scene': {
                'block_start': list(scenario.block_start),
                'block_yaw': scenario.block_yaw,
                'block_friction': scenario.block_friction,
            },
        })
        # 失败时策略处于哪个操作阶段（依据失败类型与是否抬起工件推断）
        if result['success']:
            result['failure_stage'] = 'completed'
        elif not closed_once:
            result['failure_stage'] = 'approach/grasp'
        elif result['failure'] == 'object_slipped':
            result['failure_stage'] = 'lift/transport'
        else:
            result['failure_stage'] = 'place/release'

        if args.dump_actions:
            result['actions'] = actions
        if args.save_dir:
            save_dir = Path(args.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(save_dir / f'episode_{seed:04d}.npz',
                                actions=np.asarray(actions, dtype=np.float32),
                                states=np.asarray(states, dtype=np.float32))
            from PIL import Image

            Image.fromarray(camera.capture()['top']).save(save_dir / f'episode_{seed:04d}_final_top.png')
            if video:
                video_dir = save_dir / 'videos'
                video_dir.mkdir(parents=True, exist_ok=True)
                tag = 'ok' if result['success'] else f'fail_{result["failure"]}'
                _write_video(video_dir / f'episode_{seed:04d}_{tag}.mp4', video, args.video_fps)
        return result
    finally:
        sim.close()


def run_expert_episode(seed: int, args) -> dict:
    """在同一批随机种子上跑解析式专家策略，作为 VLA 的对照基线。"""
    from robot_controller import RobotController
    from vla.expert import VLAExpertTask

    scenario = scenario_for_seed(seed, args.jitter, args.friction_jitter)
    sim, block = build_scene(scenario)
    try:
        camera = CameraRig(sim)
        for _ in range(args.warmup_steps):
            sim.step()
        task = VLAExpertTask(sim, RobotController(sim), block,
                             params=scenario.params, camera=camera)
        start = time.perf_counter()
        success = bool(task.run())
        result = task.result()
        result.update({
            'seed': int(seed),
            'ticks': int(task.transition_count()),
            'wall_seconds': round(time.perf_counter() - start, 3),
            'mean_tracking_error_rad': 0.0,
        })
        if success != bool(result.get('success')):
            result['success'] = success
        if args.save_dir:
            save_dir = Path(args.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            _, states, actions, _ = task.episode_arrays()
            np.savez_compressed(save_dir / f'episode_{seed:04d}.npz',
                                actions=np.asarray(actions, dtype=np.float32),
                                states=np.asarray(states, dtype=np.float32))
        return result
    finally:
        sim.close()


# --------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description='SmolVLA 闭环评测')
    parser.add_argument('--policy', help='checkpoint 目录或 HF repo id')
    parser.add_argument('--expert', action='store_true', help='跑解析式专家策略作为对照基线')
    parser.add_argument('--replay-dataset', help='用数据集动作回放（调试评测管线）')
    parser.add_argument('--replay-episode', type=int, default=0)
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--seed-base', type=int, default=1000)
    parser.add_argument('--max-ticks', type=int, default=260)
    parser.add_argument('--warmup-steps', type=int, default=30)
    parser.add_argument('--jitter', type=float, default=0.008)
    parser.add_argument('--friction-jitter', type=float, default=0.2)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--n-action-steps', type=int,
                        help='每个 chunk 实际执行的步数，默认沿用 checkpoint 配置（50）')
    parser.add_argument('--out', required=True, help='评测结果目录')
    parser.add_argument('--save-dir', help='逐 episode 保存动作/画面，默认 results 下')
    parser.add_argument('--record-video', action='store_true', help='录制每个 episode 的俯视视频')
    parser.add_argument('--video-fps', type=int, default=20)
    parser.add_argument('--dump-actions', action='store_true')
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_dir:
        args.save_dir = str(Path(args.save_dir))
    elif args.policy:
        args.save_dir = str(out_dir / 'episodes')
    args.dump_actions = bool(args.dump_actions) or bool(args.policy)

    if args.replay_dataset:
        runner = ExpertReplayRunner(args.replay_dataset, args.replay_episode)
        args.save_dir = args.save_dir or str(out_dir / 'episodes')
    elif args.expert:
        runner = None
        args.save_dir = str(out_dir / 'episodes')
    else:
        runner = SmolVLARunner(args.policy, args.device, n_action_steps=args.n_action_steps)

    results = []
    for i in range(args.episodes):
        seed = args.seed_base + i
        res = run_expert_episode(seed, args) if args.expert else run_episode(runner, seed, args)
        results.append(res)
        flag = 'OK  ' if res['success'] else 'FAIL'
        print(f"[{i + 1}/{args.episodes}] seed={seed} {flag} "
              f"failure={res['failure'] or '-'} lift={res['lift_height_m']:.3f} "
              f"xy_err={res['place_xy_error_m']:.3f} ticks={res['ticks']}", flush=True)

    n = len(results)
    n_ok = sum(1 for r in results if r['success'])
    failures = {}
    for r in results:
        if not r['success']:
            failures[r['failure']] = failures.get(r['failure'], 0) + 1
    summary = {
        'policy': 'expert' if args.expert else args.policy,
        'n_action_steps': args.n_action_steps,
        'episodes': n,
        'success': n_ok,
        'success_rate': round(n_ok / max(n, 1), 4),
        'failure_counts': failures,
        'failure_text': {k: FAIL_TEXT.get(k, k) for k in failures},
        'mean_lift_height_m': float(np.mean([r['lift_height_m'] for r in results])),
        'mean_place_xy_error_m': float(np.nanmean([r['place_xy_error_m'] for r in results])),
        'mean_ticks': float(np.mean([r['ticks'] for r in results])),
        'mean_tracking_error_rad': float(np.mean([r['mean_tracking_error_rad'] for r in results])),
        'mean_inference_latency_ms': float(np.mean([r.get('inference_latency_ms_mean', 0.0)
                                                   for r in results])),
        'failure_stage_counts': {
            stage: sum(1 for r in results if r.get('failure_stage') == stage)
            for stage in ('approach/grasp', 'lift/transport', 'place/release')
        },
        'seeds': [r['seed'] for r in results],
    }
    (out_dir / 'eval.json').write_text(
        json.dumps({'summary': summary, 'episodes': results}, ensure_ascii=False, indent=2),
        encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if n_ok == n else 0


if __name__ == '__main__':
    raise SystemExit(main())
