"""Why does the HOME-recovery case for C report tracking_error?

Headless diagnostic only: reuses the recorded 0B trials, rebuilds the exact
pre-successor state, then reports per-joint tracking, motor effort saturation
and whether the residual closes when the same target is held much longer.
It does not modify the controller or the recorded results.
"""
from pathlib import Path
import json
import sys
import numpy as np
import pybullet as p

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from simulation import Simulation, HOME
from robot_controller import RobotController
from skills import Z1, Z2, skill_success
from visualize_experiment_0b import load_record

NAMES = ['j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'j7']


SKILLS = {'z1': Z1, 'z2': Z2}
CASES = [('z1', 'B', 'direct'), ('z1', 'C', 'direct'), ('z2', 'B', 'direct'),
         ('z2', 'C', 'direct'), ('z1', 'C', 'reset_to_canonical')]


def rebuild(sim, ctl, implementation, mode):
    """Rebuild skill A and, when applicable, the HOME recovery."""
    skill = SKILLS[implementation]
    a_result = ctl.move_pose(skill.position, skill.orientation)
    if not (a_result.success and skill_success(sim, skill)):
        raise RuntimeError(f'skill A ({implementation}) did not succeed; cannot rebuild the state')
    if mode == 'reset_to_canonical':
        return ctl.move(np.asarray(HOME, dtype=float))
    return None


def apply_hold(sim, target, steps, tolerance):
    """Hold the commanded target and report how the error evolves."""
    history = []
    for k in range(steps):
        sim.command(target)
        sim.step()
        error = float(np.linalg.norm(sim.q() - target))
        history.append(error)
        if error < tolerance:
            return k, history
    return steps, history


def report(sim, ctl, implementation, successor, mode, verbose):
    record = load_record(implementation, successor, mode)
    target = np.asarray(json.loads(record['ik_candidate_q_rad']), dtype=float)
    rebuild(sim, ctl, implementation, mode)
    start = sim.q().copy()
    result = ctl.move(target)
    final = sim.q().copy()
    forces = [p.getJointInfo(sim.robot, i, physicsClientId=sim.client)[10] for i in range(7)]
    errors = final - target
    worst = int(np.argmax(np.abs(errors)))
    torque = p.getJointState(sim.robot, worst, physicsClientId=sim.client)[3]
    name = f'{implementation}->{successor}' + ('@reset' if mode == 'reset_to_canonical' else '')
    print(f'{name:12} {result.reason:22} L2={np.linalg.norm(errors):.3f} rad  '
          f'worst={NAMES[worst]} {errors[worst]:+.3f} rad  '
          f'torque={torque:+.2f}/{forces[worst]:.1f} N*m ({abs(torque) / forces[worst]:.0%} of limit)')
    if verbose:
        print(f'  start  {np.round(start, 3)}')
        print(f'  target {np.round(target, 3)}  |delta|={np.linalg.norm(target - start):.3f} rad')
        print(f'  {"joint":6}{"target":>9}{"final":>9}{"err":>9}{"torque":>9}{"max":>8}{"usage":>9}')
        for i in range(7):
            joint_torque = p.getJointState(sim.robot, i, physicsClientId=sim.client)[3]
            print(f'  {NAMES[i]:6}{target[i]:9.3f}{final[i]:9.3f}{errors[i]:9.3f}'
                  f'{joint_torque:9.2f}{forces[i]:8.1f}{abs(joint_torque) / forces[i]:8.0%}')
        print(f'  joint-limit violation: {bool(np.any(final < sim.lower - 1e-3) or np.any(final > sim.upper + 1e-3))}'
              f'; collisions at rest: {bool(sim.collisions())}')
        steps, history = apply_hold(sim, target, 3000, 0.06)
        print(f'  holding the same target {steps} more steps ({steps * sim.dt:.2f} s): '
              f'error {history[0]:.3f} -> {history[-1]:.3f} rad, converged={history[-1] < 0.06}')
        held = p.getJointState(sim.robot, 4, physicsClientId=sim.client)[3]
        print(f'  j5 torque after settling: {held:+.2f}/{forces[4]:.1f} N*m '
              f'({abs(held) / forces[4]:.0%} of limit)')
    return result


def main():
    print(f'{"case":12} {"reason":22} tracking summary')
    for implementation, successor, mode in CASES:
        verbose = (implementation, successor, mode) == ('z1', 'C', 'reset_to_canonical')
        sim = Simulation(gui=False)
        ctl = RobotController(sim)
        try:
            report(sim, ctl, implementation, successor, mode, verbose)
        finally:
            sim.close()


if __name__ == '__main__':
    main()
