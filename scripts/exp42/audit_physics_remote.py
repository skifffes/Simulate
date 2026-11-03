"""Isolated eight-step teacher replay and paired action-contract probes."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = ['/scratch/lijh/exp40/repo', '/scratch/lijh/exp40/repo/src']
    import pybullet as p
    from src.exp41.scene41 import build_a_scene, build_b_scene, A_GOALS, B_OBJECT_STARTS, B_ROBOT_STARTS
    from types import SimpleNamespace
    from src.exp40.action_adapter import chunk_to_targets, ActionAdapterConfig
    data = Path('/scratch/lijh/exp42/data')
    metadata = [json.loads(x) for x in (data / 'meta.jsonl').read_text().splitlines() if x]
    samples = np.load(args.samples, allow_pickle=False)
    results = []
    for skill in ['A90', 'B90']:
        row = next(r for r in metadata if r['skill'] == skill and r['split'] == 'val')
        with np.load(data / row['state_path']) as z:
            states, actions = z['state'], z['action']
        for phase, frame in [('startup', 0), ('approach', 8), ('middle_temporal', len(states)//2), ('late_temporal', 3*len(states)//4)]:
            frame = min(frame, len(states)-16)
            predictions = samples[f'{skill}_{phase}_t{frame}_prediction']
            for mode in ['teacher', 'original', 'anchor_only', 'gripper_only', 'fixed_anchor_binary', 'fixed_anchor_continuous']:
                for seed_index in ([0] if mode == 'teacher' else range(len(predictions))):
                    base_scene = (build_a_scene(A_GOALS[4], seed=94000000) if skill == 'A90' else
                                  build_b_scene(B_OBJECT_STARTS[1], B_ROBOT_STARTS[0], seed=94000000))
                    scene = SimpleNamespace(robot_id=base_scene.robot, client_id=base_scene.sim.client,
                        object_id=base_scene.block, arm_joints=list(range(7)), finger_joints=[9, 10], ee_link=11)
                    q0 = states[frame, :7].copy()
                    for joint, value in zip(scene.arm_joints, q0):
                        p.resetJointState(scene.robot_id, joint, float(value), physicsClientId=scene.client_id)
                    for joint, value in zip(scene.finger_joints, states[frame, 7:9]):
                        p.resetJointState(scene.robot_id, joint, float(value), physicsClientId=scene.client_id)
                    def ee():
                        return np.array(p.getLinkState(scene.robot_id, scene.ee_link, computeForwardKinematics=True,
                                                      physicsClientId=scene.client_id)[4])
                    start_ee = ee()
                    initial_object = p.getBasePositionAndOrientation(scene.object_id, physicsClientId=scene.client_id)[0]
                    initial_contacts = len(p.getContactPoints(scene.robot_id, scene.object_id, physicsClientId=scene.client_id))
                    if mode != 'teacher':
                        pred = predictions[seed_index]
                        cfg = ActionAdapterConfig(gripper_mode='continuous' if mode.endswith('continuous') else 'binary')
                        if mode in {'original', 'gripper_only'}:
                            adapted = chunk_to_targets(pred[:8], q0, base_scene.sim.lower, base_scene.sim.upper, cfg)
                        else:
                            adapted = [chunk_to_targets(pred[i:i+1], q0, base_scene.sim.lower,
                                                       base_scene.sim.upper, cfg)[0] for i in range(8)]
                    for step in range(8):
                        if mode == 'teacher':
                            target = actions[frame+step, :7]
                            width = float(actions[frame+step, 7])
                        else:
                            target = adapted[step].q_target
                            width = float(adapted[step].gripper_value if mode in {'original', 'anchor_only'} else adapted[step].gripper_target_m)
                        before = np.array([p.getJointState(scene.robot_id, j, physicsClientId=scene.client_id)[0] for j in scene.arm_joints])
                        before_ee = ee()
                        base_scene.sim.command(np.asarray(target, dtype=np.float64))
                        for j in scene.finger_joints:
                            p.setJointMotorControl2(scene.robot_id, j, p.POSITION_CONTROL, targetPosition=width,
                                force=30.0, physicsClientId=scene.client_id)
                        for _ in range(12):
                            base_scene.sim.step()
                        joint_state = [p.getJointState(scene.robot_id, j, physicsClientId=scene.client_id) for j in scene.arm_joints]
                        after = np.array([x[0] for x in joint_state])
                        after_ee = ee()
                        results.append({'skill': skill, 'episode': row['episode_index'], 'frame': frame, 'phase': phase,
                            'mode': mode, 'sample_index': seed_index, 'step': step, 'target_q': json.dumps(target.tolist()),
                            'q_before': json.dumps(before.tolist()), 'q_after': json.dumps(after.tolist()),
                            'qd_after': json.dumps([x[1] for x in joint_state]), 'finger_target_m': width,
                            'target_error_max_rad': float(np.abs(target-before).max()),
                            'actual_delta_max_rad': float(np.abs(after-before).max()),
                            'ee_before': json.dumps(before_ee.tolist()), 'ee_after': json.dumps(after_ee.tolist()),
                            'ee_displacement_from_start_m': float(np.linalg.norm(after_ee-start_ee)),
                            'contacts': len(p.getContactPoints(scene.robot_id, scene.object_id, physicsClientId=scene.client_id)),
                            'initial_contacts': initial_contacts, 'initial_object_position': json.dumps(initial_object)})
                    p.disconnect(scene.client_id)
    with (args.out / 'physical_contract_trace.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    (args.out / 'physics_limits.json').write_text(json.dumps({'rows': len(results),
        'scope': 'Eight-step isolated robot-state replay only; no full skill success claim.',
        'limits': ['Object scene reset to canonical initial pose, not phase-matched demonstration object pose.',
                   'Middle/late temporal strata are NOT contact-phase evidence.',
                   'Original vs fixed_anchor_binary changes both anchor reconstruction and gripper units; not a single-variable ablation.',
                   'fixed_anchor_binary vs fixed_anchor_continuous isolates gripper mapping.']}, indent=2))
    print(json.dumps({'rows': len(results), 'out': str(args.out)}))


if __name__ == '__main__':
    main()
