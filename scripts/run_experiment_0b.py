"""Experiment 0B: multi-start geometric sensitivity, not VLA generalization."""
from pathlib import Path
import argparse
import csv
import json
import sys
import hashlib
import platform
import shutil
from datetime import datetime, timezone
import numpy as np
import pybullet as p
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from simulation import Simulation, HOME
from robot_controller import RobotController, wrap_into_limits
from skills import Z1, Z2, skill_success

OUT = ROOT / 'results' / 'experiment_0b'
SEED = 20261020
TRIALS = 3
# Fixed in the experiment configuration before execution. B and C differ in both
# Cartesian position and orientation. These are geometry probes, not object actions.
TARGETS = {
    'B': {'position': (0.55, 0.0, 0.38), 'euler': (np.pi, 0.0, np.pi / 2)},
    'C': {'position': (0.55, -0.20, 0.16), 'euler': (np.pi / 2, 0.0, 0.0)},
}
START_OFFSETS = [np.zeros(7), np.array([0.18, -0.12, 0.16, 0.12, -0.15, 0.10, 0.20]),
                 np.array([-0.18, 0.12, -0.16, -0.12, 0.15, -0.10, -0.20]),
                 np.array([0.0, 0.20, 0.0, -0.18, 0.0, 0.12, -0.15])]


def target_pose(name):
    item = TARGETS[name]
    return np.asarray(item['position'], dtype=float), p.getQuaternionFromEuler(item['euler'])


def collision_free_path(sim, start, goal, samples=80):
    """Probe interpolated configurations without moving the simulated robot state."""
    saved = sim.q()
    try:
        for alpha in np.linspace(0.0, 1.0, samples + 1):
            q = start + alpha * (goal - start)
            for joint, value in zip(sim.joints, q):
                p.resetJointState(sim.robot, joint, float(value), physicsClientId=sim.client)
            p.performCollisionDetection(physicsClientId=sim.client)
            if sim.collisions():
                return False, float(alpha)
        return True, None
    finally:
        for joint, value in zip(sim.joints, saved):
            p.resetJointState(sim.robot, joint, float(value), physicsClientId=sim.client)
        sim.command(saved)
        p.performCollisionDetection(physicsClientId=sim.client)


def run_one(implementation, successor, trial, seed, mode='direct'):
    sim = Simulation(gui=False)
    ctl = RobotController(sim)
    rng = np.random.default_rng(seed)
    try:
        skill = Z1 if implementation == 'z1' else Z2
        a_result = ctl.move_pose(skill.position, skill.orientation)
        a_ok = bool(a_result.success and skill_success(sim, skill))
        start = sim.q().copy()
        if mode == 'reset_to_canonical' and a_ok:
            home_motion = ctl.move(np.asarray(HOME, dtype=float))
            reset_success = bool(home_motion.success and np.allclose(sim.q(), HOME, atol=0.04))
        else:
            reset_success = True
        start = sim.q().copy()
        execution_state = p.saveState(physicsClientId=sim.client)
        reset_seconds = home_motion.seconds if mode == 'reset_to_canonical' and a_ok else 0.0
        reset_distance = home_motion.distance if mode == 'reset_to_canonical' and a_ok else 0.0
        target_position, target_orientation = target_pose(successor)
        rows = []
        for start_index, offset in enumerate(START_OFFSETS):
            p.restoreState(execution_state, physicsClientId=sim.client)
            # Truly set each seed configuration in simulation before calling IK.
            seed_q = np.clip(start + offset + rng.normal(0, 0.015, 7), sim.lower + 1e-4, sim.upper - 1e-4)
            for joint, value in zip(sim.joints, seed_q):
                p.resetJointState(sim.robot, joint, float(value), physicsClientId=sim.client)
            p.performCollisionDetection(physicsClientId=sim.client)
            returned = p.calculateInverseKinematics(sim.robot, sim.ee, target_position.tolist(),
                np.asarray(target_orientation).tolist(), lowerLimits=sim.lower.tolist(), upperLimits=sim.upper.tolist(),
                jointRanges=(sim.upper-sim.lower).tolist(), restPoses=seed_q.tolist(),
                maxNumIterations=300, residualThreshold=1e-5, physicsClientId=sim.client)
            candidate, wrapped = wrap_into_limits(returned[:7], sim.lower, sim.upper, seed_q)
            limit_ok = bool(np.all(candidate >= sim.lower - 1e-5) and np.all(candidate <= sim.upper + 1e-5))
            for joint, value in zip(sim.joints, candidate):
                p.resetJointState(sim.robot, joint, float(value), physicsClientId=sim.client)
            pos, quat = sim.pose()
            pe = float(np.linalg.norm(pos - target_position))
            oe = float(2 * np.arccos(np.clip(abs(np.dot(quat, target_orientation)), 0, 1)))
            ik_ok = bool(limit_ok and np.isfinite(pe) and pe < 0.025 and oe < 0.15)
            row = {'implementation': implementation, 'successor': successor, 'mode': mode, 'trial': trial,
                   'seed': seed, 'ik_start_index': start_index, 'a_success': a_ok,
                   'reset_success': reset_success, 'start_q_rad': json.dumps(start.tolist()),
                   'reset_seconds': reset_seconds, 'reset_distance_rad': reset_distance,
                   'ik_seed_q_rad': json.dumps(seed_q.tolist()), 'ik_raw_q_rad': json.dumps(list(returned[:7])),
                   'ik_joint_limits_ok': limit_ok, 'ik_wrapped': bool(np.any(wrapped)),
                   'ik_pose_ok': ik_ok, 'ik_position_error_m': pe, 'ik_orientation_error_rad': oe,
                   'ik_candidate_q_rad': json.dumps(candidate.tolist()), 'path_feasible': False,
                   'path_collision_alpha': None, 'executed': False, 'execution_success': False,
                   'failure_reason': 'ik_joint_limits' if not limit_ok else ('ik_pose_residual' if not ik_ok else None),
                   'actual_position_error_m': None, 'actual_orientation_error_rad': None,
                   'actual_joint_tracking_error_rad': None,
                   'trajectory_length_rad': None, 'execution_seconds': None,
                   'total_handoff_seconds': None, 'total_handoff_distance_rad': None}
            if ik_ok:
                path_ok, collision_alpha = collision_free_path(sim, start, candidate)
                row['path_feasible'] = path_ok
                row['path_collision_alpha'] = collision_alpha
                if not path_ok:
                    row['failure_reason'] = 'path_collision'
                if path_ok and a_ok and reset_success:
                    p.restoreState(execution_state, physicsClientId=sim.client)
                    sim.command(start)
                    motion = ctl.move(candidate)
                    pos, quat = sim.pose()
                    actual_joint_error = float(np.linalg.norm(sim.q() - candidate))
                    actual_pe = float(np.linalg.norm(pos - target_position))
                    actual_oe = float(2 * np.arccos(np.clip(abs(np.dot(quat, target_orientation)), 0, 1)))
                    success = bool(motion.success and actual_joint_error < 0.06 and actual_pe < 0.025 and actual_oe < 0.15)
                    reason = None if success else (motion.reason if not motion.success else 'execution_tracking_or_cartesian_error')
                    row.update(executed=True, execution_success=success, failure_reason=reason,
                               actual_position_error_m=actual_pe, actual_orientation_error_rad=actual_oe,
                               actual_joint_tracking_error_rad=actual_joint_error,
                               trajectory_length_rad=motion.distance, execution_seconds=motion.seconds,
                               total_handoff_seconds=reset_seconds + motion.seconds,
                               total_handoff_distance_rad=reset_distance + motion.distance,
                               reset_seconds=reset_seconds if mode == 'reset_to_canonical' else 0.0,
                               reset_distance_rad=reset_distance if mode == 'reset_to_canonical' else 0.0)
            if not a_ok:
                row['failure_reason'] = 'a_failed:' + a_result.reason
            elif not reset_success:
                row['failure_reason'] = 'reset_to_canonical_failed'
            elif row['executed'] and not row['execution_success'] and row['failure_reason'] is None:
                row['failure_reason'] = 'cartesian_pose_error'
            rows.append(row)
        return rows
    finally:
        sim.close()


def hash_existing_inputs():
    paths = [ROOT / 'results' / 'results.csv', ROOT / 'results' / 'summary.csv']
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths if path.exists()}


def make_plot(summary, output):
    width, height = 1100, 620
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype('arial.ttf', 20)
        small = ImageFont.truetype('arial.ttf', 15)
    except OSError:
        font, small = ImageFont.load_default(), ImageFont.load_default()
    draw.text((35, 20), 'Experiment 0B: geometric handoff cost', fill='black', font=font)
    groups = {}
    for row in summary:
        implementation = row['condition'].split('->')[0]
        key = (row['mode'], implementation)
        groups.setdefault(key, {})[row['successor'] if 'successor' in row else row['condition'].split('->')[1]] = row['mean_successful_length_rad']
    colors = {'direct': '#3578b8', 'reset_to_canonical': '#e68632'}
    y_positions = [125, 190, 300, 365]
    labels = [('direct', 'z1'), ('direct', 'z2'), ('reset_to_canonical', 'z1'), ('reset_to_canonical', 'z2')]
    maximum = max([v for pair in groups.values() for v in pair.values() if v is not None] or [1])
    for y, (mode, implementation) in zip(y_positions, labels):
        row = groups.get((mode, implementation), {})
        draw.text((35, y), f'{mode.replace("_", " ")} {implementation}', fill='black', font=small)
        for j, successor in enumerate(('B', 'C')):
            value = row.get(successor)
            x0, yy = 260, y + j * 22
            draw.text((195, yy), successor, fill='black', font=small)
            if value is not None:
                draw.rectangle((x0, yy, x0 + int(600 * value / maximum), yy + 16), fill=colors[mode])
                draw.text((x0 + int(600 * value / maximum) + 8, yy), f'{value:.3f} rad', fill='black', font=small)
            else:
                draw.text((x0, yy), 'No successful execution (see CSV)', fill='#b02020', font=small)
    draw.text((35, 470), 'B and C have different positions AND orientations; compare B/C cost ranking within z1/z2.', fill='black', font=small)
    draw.text((35, 500), 'Discrete collision probing is not a continuous collision/path-planning proof.', fill='black', font=small)
    draw.text((35, 530), 'Geometry sanity check only; this is not evidence of VLA compositional generalization.', fill='black', font=small)
    image.save(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=TRIALS)
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error('--trials must be positive')
    if OUT.exists() and any(OUT.iterdir()):
        archive = ROOT / 'results' / 'experiment_0b_history' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        shutil.copytree(OUT, archive)
        print(f'Previous 0B outputs archived: {archive}')
    OUT.mkdir(parents=True, exist_ok=True)
    before_hashes = hash_existing_inputs()
    rows = []
    for mode in ('direct', 'reset_to_canonical'):
        for implementation in ('z1', 'z2'):
            for successor in ('B', 'C'):
                for trial in range(args.trials):
                    seed = args.seed + trial
                    rows.extend(run_one(implementation, successor, trial, seed, mode))
    fields = list(rows[0])
    with (OUT / 'multistart_results.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for mode in ('direct', 'reset_to_canonical'):
        for implementation in ('z1', 'z2'):
            for successor in ('B', 'C'):
                group = [r for r in rows if r['mode'] == mode and r['implementation'] == implementation and r['successor'] == successor]
                solved = [r for r in group if r['ik_joint_limits_ok']]
                feasible = [r for r in group if r['path_feasible']]
                executed = [r for r in group if r['execution_success']]
                summary.append({'mode': mode, 'condition': f'{implementation}->{successor}',
                    'ik_pose_valid_candidates': sum(r['ik_pose_ok'] for r in group),
                    'ik_candidates_in_limits': len(solved), 'ik_candidates_total': len(group),
                    'mean_successful_total_handoff_s': float(np.mean([r['total_handoff_seconds'] for r in executed])) if executed else None,
                    'mean_successful_total_handoff_rad': float(np.mean([r['total_handoff_distance_rad'] for r in executed])) if executed else None,
                    'path_feasible_candidates': len(feasible), 'execution_successes': len(executed),
                    'execution_attempts': sum(r['executed'] for r in group),
                    'mean_successful_length_rad': float(np.mean([r['trajectory_length_rad'] for r in executed])) if executed else None,
                    'mean_successful_time_s': float(np.mean([r['execution_seconds'] for r in executed])) if executed else None,
                    'deterministic_repetitions_not_independent_samples': True})
    with (OUT / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    after_hashes = hash_existing_inputs()
    config = {'experiment': '0B geometric sensitivity sanity check', 'seed': args.seed,
        'trials_per_condition': args.trials, 'ik_starts_per_trial': len(START_OFFSETS),
        'fixed_timestep_s': 1/240, 'home_joint_positions_rad': HOME,
        'targets': {name: {'position_m': list(spec['position']), 'euler_rad': list(spec['euler'])} for name, spec in TARGETS.items()},
        'path_check': {'method': '80-segment joint-space interpolation; temporary resetJointState for collision probing only',
                       'execution': 'motor-controlled RobotController.move after path probe'},
        'modes': ['direct', 'reset_to_canonical'],
        'exploration_disclosure': 'Before this fixed configuration was established, exploratory tests had been run for C at (0.55,-0.20,0.16), including Euler orientations (pi/2,0,0), (pi/2,0,pi/2), and (pi/2,0,-pi/2), plus a nearby position. Interpret as exploratory sanity check, not preregistered confirmation.',
        'scope': 'Geometric/kinematic sensitivity test only; not evidence of VLA compositional generalization.',
        'repetitions_note': 'Deterministic repeated runs are not statistically independent physics trials.',
        'physics_collision_artifacts': 'The Panda URDF has a persistent ~2.5 cm overlap between panda_link7 (link 6) and panda_hand (link 8). Failure detection already excluded that pair, but PyBullet still solved the contact, braking panda_joint5 at its 12 N*m effort limit. That produced the earlier HOME->C tracking_error. The pair is now disabled in dynamics as well (Simulation.ignored_contacts); the pre-fix run is archived in results/experiment_0b_history/.',
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'pybullet': getattr(p, '__version__', 'unknown')},
        'generated_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'ik_seed_offsets_rad': [offset.tolist() for offset in START_OFFSETS],
        'ik_seed_noise_std_rad': 0.015, 'ik_max_iterations': 300,
        'ik_residual_threshold': 1e-5,
        'acceptance': {'position_m': 0.025, 'orientation_rad': 0.15,
                       'joint_tracking_rad': 0.06},
        'pre_existing_data_sha256_before': before_hashes, 'pre_existing_data_sha256_after': after_hashes,
        'pre_existing_data_unchanged': before_hashes == after_hashes}
    (OUT / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    make_plot(summary, OUT / 'comparison.png')
    (OUT / 'README.md').write_text(
        '# 实验 0B：后继任务几何敏感性检查\n\n'
        '本实验让 A 分别产生 z1/z2 两种终态，再执行位置、朝向都不同的 B/C 末端目标。每个条件使用 4 个关节初值做 IK，多起点结果先做限位检查，再做 80 段离散关节插值碰撞采样，路径通过后由原电机控制器执行。保留 direct 与 Reset-to-Canonical 对照。\n\n'
        '这只是几何与控制器 sanity check，不是物体交接实验，也不是 VLA 组合泛化证据。路径碰撞检查是离散采样，不构成连续路径安全证明。重复运行确定性场景不等于独立统计样本。\n\n'
        '运行：`python scripts/run_experiment_0b.py`。结果在本目录：`multistart_results.csv`（逐起点原始记录）、`summary.csv`、`comparison.png`、`config.json`（参数、种子、探索披露、旧 CSV SHA-256）。不覆盖原实验结果。\n', encoding='utf-8')
    report = [
        '\n## 本次结果\n',
        '|模式|条件|成功候选/尝试|后继行程 rad|后继时间 s|',
        '|---|---|---:|---:|---:|',
    ]
    for item in summary:
        distance = item['mean_successful_length_rad']
        seconds = item['mean_successful_time_s']
        cost = f'{distance:.3f}|{seconds:.3f}' if distance is not None else '—|—'
        report.append(f"|{item['mode']}|{item['condition']}|{item['execution_successes']}/{item['execution_attempts']}|{cost}|")
    report.extend([
        '\n注意：每组 12 条是 3 个 IK 随机种子 × 4 个初始猜测，不是 12 次独立物理试验。物理环境未随机化，IK 初值含标准差 0.015 rad 的扰动。',
        '以上成本是所有成功候选的均值，不是挑选最快候选；各候选从同一动力学快照独立执行。汇总 CSV 另有包含恢复动作的总成本。',
        '此前的 HOME→C tracking_error 已定位为仿真接触伪影：panda_link7 与 panda_hand 的碰撞几何常年重叠约 2.5 cm，物理引擎仍求解该接触，给第 5 关节附加约 12 N·m 制动使其饱和。该固定连接对现已在动力学层关闭，相关旧结果归档在 results/experiment_0b_history/；关闭后 HOME→C 全部成功。',
        '多起点仅检查交接状态附近 4 个初值，不是全局 IK 或路径规划证明。path_feasible 仅指当前碰撞检测器对 80 段插值的检查结果。',
        '此前探索过多个 C 朝向及邻近位置，详见 config.json；本次是探索性结果，不是预注册确认实验。后续改变参数必须保留版本，不能只报告出现交叉的配置。',
        '阶段二方案见 docs/experiment_0b_stage2_design.md；仅设计，尚未实现或验证物体操作。',
    ])
    with (OUT / 'README.md').open('a', encoding='utf-8') as handle:
        handle.write('\n'.join(report) + '\n')
    if before_hashes != after_hashes:
        raise RuntimeError('Original experiment CSV changed during 0B run')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
