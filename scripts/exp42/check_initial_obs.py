"""检查：闭环 rollout 的**第一步观测**是否与数据集 episode 的起始帧一致。

如果 rollout 起始状态/图像与训练数据差异很大，那么"初始帧动作错误"可能只是分布外，
而不是模型没学会。本脚本对比：
  * 关节 q（7 维）与夹爪归一化值；
  * 外部/腕部相机图像的均值/标准差；
  * rollout 使用 scene41.build_a_scene(seed=9000)（与 eval_rollouts 相同）。

用法（simulate venv）：
  python scripts/exp42/check_initial_obs.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / 'src', ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

DATA = Path(os.environ.get('EXP42_DATA', '/scratch/lijh/exp42/data'))
FINGER_MAX_M = 0.04


def main() -> int:
    import pybullet as p

    from exp41 import scene41
    from vla.cameras import CameraRig

    meta = [json.loads(l) for l in (DATA / 'meta.jsonl').read_text(encoding='utf-8').splitlines()
            if l.strip()]
    a90 = next(m for m in meta if m['skill'] == 'A90' and m['split'] == 'train')
    sa = np.load(DATA / a90['state_path'])
    st_db = sa['state'][0].astype(float)
    top_db = np.load(DATA / a90['top_path'], mmap_mode='r')[0]
    wrist_db = np.load(DATA / a90['wrist_path'], mmap_mode='r')[0]

    goal = next(g for g in scene41.A_GOALS if g.goal_id == 'A_x00_y00')
    scene = scene41.build_a_scene(goal, seed=9000)
    rig = CameraRig(scene.sim, scene.cameras, (256, 256), supersample=1)
    frames = rig.capture()
    q = np.asarray(scene.sim.q(), dtype=float)
    fingers = np.array([p.getJointState(scene.sim.robot, j, physicsClientId=scene.sim.client)[0]
                        for j in (9, 10)])
    grip = float(np.clip(fingers.mean() / FINGER_MAX_M, 0, 1))
    top = np.asarray(frames['top'], dtype=np.uint8)
    wrist = np.asarray(frames['wrist'], dtype=np.uint8)

    res = {
        'dataset_episode': a90['episode_index'],
        'q_rollout': q.round(5).tolist(),
        'q_dataset_frame0': st_db[:7].round(5).tolist(),
        'q_max_abs_diff': float(np.abs(q - st_db[:7]).max()),
        'gripper_rollout': grip,
        'gripper_dataset_frame0': float(np.clip(st_db[7:9].mean() / FINGER_MAX_M, 0, 1)),
        'image_top_mean_rollout': float(top.mean()), 'image_top_mean_dataset': float(top_db.mean()),
        'image_top_std_rollout': float(top.std()), 'image_top_std_dataset': float(top_db.std()),
        'image_wrist_mean_rollout': float(wrist.mean()),
        'image_wrist_mean_dataset': float(wrist_db.mean()),
        'image_top_shape': [list(top.shape), list(top_db.shape)],
    }
    # 渲染差：同一场景、同一相机下，rollout 的图 vs 数据集的图（像素级）
    res['image_top_absdiff_mean'] = float(np.abs(top.astype(float) - top_db.astype(float)).mean())
    scene.close()
    print(json.dumps(res, ensure_ascii=False, indent=2))
    out = ROOT / 'results' / 'experiment_4_2' / 'initial_obs_check.json'
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
