"""Live GUI replay of the recorded 0B trials.

Opens the PyBullet window and performs each recorded handoff in the real
simulation, with a live HUD, end-effector trail and per-case result. The joint
target is the recorded IK candidate and the original motor controller is used;
no IK is re-solved and no parameter is searched.

Keys: SPACE skip the current hold, P pause/resume, ESC quit.
"""
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
import pybullet as p

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from simulation import Simulation, HOME
from robot_controller import RobotController
from skills import Z1, Z2, skill_success
from visualize_experiment_0b import load_record, COLORS

SKILLS = {'z1': Z1, 'z2': Z2}
KEY_ESC, KEY_SPACE, KEY_P = 27, 32, 112
WHITE, GREEN, RED, GREY = (1, 1, 1), (0.15, 0.85, 0.25), (0.95, 0.2, 0.2), (0.75, 0.75, 0.75)


class Hud:
    """Several world-anchored text lines that are replaced in place."""

    def __init__(self, client, origin=(0.02, -0.55, 1.12), spacing=0.052, size=1.15):
        self.client, self.origin, self.spacing, self.size = client, origin, spacing, size
        self.ids = []

    def show(self, lines):
        while len(self.ids) < len(lines):
            self.ids.append(-1)
        lines = list(lines) + [' '] * (len(self.ids) - len(lines))
        for i, item in enumerate(lines):
            text, color = item if isinstance(item, tuple) else (item, WHITE)
            self.ids[i] = p.addUserDebugText(
                text or ' ', [self.origin[0], self.origin[1], self.origin[2] - i * self.spacing],
                textColorRGB=color, textSize=self.size, replaceItemUniqueId=self.ids[i],
                physicsClientId=self.client)


def poll_keys(sim):
    try:
        return p.getKeyboardEvents(physicsClientId=sim.client)
    except p.error:
        return {}


def pressed(events, key):
    return bool(events.get(key, 0) & p.KEY_WAS_TRIGGERED)


def quit_requested(sim):
    return pressed(poll_keys(sim), KEY_ESC)


def handle_pause(sim, state):
    """Freeze the replay without stepping physics, so the pose is held exactly."""
    while state['paused']:
        time.sleep(0.02)
        events = poll_keys(sim)
        if pressed(events, KEY_P):
            state['paused'] = False
        if pressed(events, KEY_ESC):
            state['quit'] = True
            return


def mark_target(sim, name, position, color):
    """Draw a small crosshair and a vertical drop line at a recorded target."""
    x, y, z = [float(v) for v in position]
    for dx, dy in ((0.02, 0), (-0.02, 0), (0, 0.02), (0, -0.02)):
        p.addUserDebugLine([x + dx, y + dy, z], [x + dx, y + dy, z + 0.03], color, 2.5,
                           physicsClientId=sim.client)
    p.addUserDebugLine([x, y, z], [x, y, 0], [c * 0.6 for c in color], 1.2,
                       physicsClientId=sim.client)
    p.addUserDebugText(name, [x, y, z + 0.05], textColorRGB=color, textSize=1.3,
                       physicsClientId=sim.client)


def make_observer(sim, hud, state, lines_static, phase_label, error_of):
    """Trail, HUD and pause handling; observation only, never affects control."""
    previous_q = sim.q().copy()
    previous_pos = sim.pose()[0].copy()
    traveled = {'rad': 0.0}

    def observer(k, n, q, seconds):
        handle_pause(sim, state)
        traveled['rad'] += float(np.linalg.norm(q - previous_q))
        previous_q[:] = q
        position = sim.pose()[0]
        p.addUserDebugLine(previous_pos.tolist(), position.tolist(), lines_static['color'], 3,
                           physicsClientId=sim.client)
        previous_pos[:] = position
        if k % 4 == 0:
            hud.show(list(lines_static['lines']) + [
                (f'{phase_label}  t={seconds:5.2f}s  travel={traveled["rad"]:5.3f} rad', WHITE),
                (f'{error_of(q)}  | progress {k}/{n + 60}', GREY),
            ])

    return observer, traveled


def run_move(sim, ctl, hud, state, target, lines_static, phase_label):
    """Command one joint target (HOME or a recorded IK candidate)."""
    observer, traveled = make_observer(
        sim, hud, state, lines_static, phase_label,
        lambda q: f'joint_err={float(np.linalg.norm(q - target)):.4f} rad')
    result = ctl.move(target, observer=observer)
    return result, traveled['rad']


def run_pose_move(sim, ctl, hud, state, position, orientation, lines_static, phase_label):
    """Command one Cartesian pose (skill A), solved by the original IK."""
    goal = np.asarray(position, dtype=float)
    observer, traveled = make_observer(
        sim, hud, state, lines_static, phase_label,
        lambda q: f'pos_err={float(np.linalg.norm(sim.pose()[0] - goal)) * 1000:.1f} mm')
    result = ctl.move_pose(position, orientation, observer=observer)
    return result, traveled['rad']


def hold(sim, hud, lines, seconds, state, countdown=True):
    """Keep simulating while displaying the outcome; SPACE ends the hold early."""
    total = max(1, int(seconds / sim.dt))
    for i in range(total):
        events = poll_keys(sim)
        if pressed(events, KEY_ESC):
            state['quit'] = True
            return
        if pressed(events, KEY_SPACE) and countdown:
            return
        if pressed(events, KEY_P):
            state['paused'] = not state['paused']
        handle_pause(sim, state)
        if state['quit']:
            return
        sim.step()
        if i % 24 == 0:
            remaining = (total - i) * sim.dt
            suffix = f'   hold {remaining:.1f}s (SPACE next, P pause, ESC quit)' if countdown else ''
            hud.show(list(lines) + [(suffix, GREY)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', default='all', help="'all' or e.g. z1->C, z1->C@reset")
    parser.add_argument('--speed', type=float, default=0.6, help='replay speed factor (0.5 = half speed)')
    parser.add_argument('--hold', type=float, default=3.0, help='seconds to keep each result on screen')
    parser.add_argument('--check', action='store_true', help='only validate records, do not open the GUI')
    parser.add_argument('--auto-exit', action='store_true', help='close the window when all cases finish')
    args = parser.parse_args()

    cases = [('z1', 'B', 'direct'), ('z1', 'C', 'direct'), ('z2', 'B', 'direct'),
             ('z2', 'C', 'direct'), ('z1', 'C', 'reset_to_canonical')]
    if args.case != 'all':
        selector = args.case.replace('@reset', '')
        implementation, successor = selector.split('->')
        mode = 'reset_to_canonical' if '@reset' in args.case else 'direct'
        cases = [(implementation.strip(), successor.strip(), mode)]

    records = [load_record(*case) for case in cases]
    if args.check:
        for case, record in zip(cases, records):
            print(f'{case[0]}->{case[1]} ({case[2]}): trial {record["trial"]} '
                  f'start {record["ik_start_index"]} success={record["execution_success"]} '
                  f'{float(record["trajectory_length_rad"]):.3f} rad '
                  f'{float(record["execution_seconds"]):.3f} s')
        return

    sim = Simulation(gui=True, realtime=True, speed=args.speed)
    state = {'paused': False, 'quit': False}
    hud = Hud(sim.client)
    p.resetDebugVisualizerCamera(1.55, 52, -32, [0.32, -0.06, 0.32], physicsClientId=sim.client)
    color_of = {'z1->B': COLORS['z1->B'], 'z1->C': COLORS['z1->C'],
                'z2->B': COLORS['z2->B'], 'z2->C': COLORS['z2->C'],
                'z1->C@reset': COLORS['z1->C (reset)']}
    total = len(cases)
    try:
        for index, ((implementation, successor, mode), record) in enumerate(zip(cases, records), start=1):
            name = f'{implementation}->{successor}' + ('@reset' if mode == 'reset_to_canonical' else '')
            sim.reset()
            hud = Hud(sim.client)
            p.resetDebugVisualizerCamera(1.55, 52, -32, [0.32, -0.06, 0.32], physicsClientId=sim.client)
            color = color_of[name]
            skill = SKILLS[implementation]
            mark_target(sim, 'B', np.array([0.55, 0, 0.38]), (0.95, 0.95, 0.95))
            mark_target(sim, 'C', np.array([0.55, -0.20, 0.16]), (0.55, 0.55, 0.55))
            for tag, pose in (('z1', Z1), ('z2', Z2)):
                p.addUserDebugText(tag, list(np.asarray(pose.position) + np.array([0, 0, 0.04])),
                                   textColorRGB=(0.45, 0.7, 0.95), textSize=1.15,
                                   physicsClientId=sim.client)
            header = [
                (f'[{index}/{total}] {name}   mode={mode}', color),
                (f'source: trial {record["trial"]}, ik_start {record["ik_start_index"]}, '
                 f'recorded {float(record["trajectory_length_rad"]):.3f} rad / '
                 f'{float(record["execution_seconds"]):.2f} s', GREY),
            ]
            hud.show(header + [('phase: A (' + implementation + ')', WHITE)])
            ctl = RobotController(sim)
            print(f'[{index}/{total}] {name}: phase A ({"z1" if implementation == "z1" else "z2"})')
            a_result, _ = run_pose_move(sim, ctl, hud, state, skill.position, skill.orientation,
                                        {'lines': header, 'color': color}, 'phase A')
            a_ok = bool(a_result.success and skill_success(sim, skill))
            if state['quit']:
                return
            if not a_ok:
                print(f'  A failed: {a_result.reason}')
                hold(sim, hud, header + [(f'STOP: A failed ({a_result.reason})', RED)],
                     args.hold, state)
                continue
            if mode == 'reset_to_canonical':
                hud.show(header + [('phase: recovery to HOME', WHITE)])
                print('  recovery: HOME')
                run_move(sim, ctl, hud, state, np.asarray(HOME, dtype=float),
                         {'lines': header, 'color': (0.9, 0.75, 0.2)}, 'HOME')
                if state['quit']:
                    return
            target = np.asarray(json.loads(record['ik_candidate_q_rad']), dtype=float)
            hud.show(header + [(f'phase: successor {successor}', WHITE)])
            print(f'  successor {successor}')
            result, traveled = run_move(sim, ctl, hud, state, target,
                                        {'lines': header, 'color': color}, f'successor {successor}')
            if state['quit']:
                return
            position, quat = sim.pose()
            joint_error = float(np.linalg.norm(sim.q() - target))
            recorded_ok = record['execution_success'] == 'True'
            print(f'  result: {result.success} ({result.reason}); {traveled:.3f} rad, '
                  f'{result.seconds:.3f} s, joint_err {joint_error:.3f}')
            outcomes = header + [
                (f'successor {successor}: {traveled:.3f} rad, {result.seconds:.2f} s, '
                 f'joint_err {joint_error:.3f}', WHITE),
                ('SUCCESS' if result.success else f'FAILED: {result.reason}', GREEN if result.success else RED),
                (f'recorded outcome: {"SUCCESS" if recorded_ok else "tracking_error"}', GREY),
            ]
            hold(sim, hud, outcomes, args.hold, state)
            if state['quit']:
                return
        if args.auto_exit:
            return
        hud.show([('all cases finished', WHITE),
                  ('close the window or press ESC to exit', GREY)])
        print('all cases finished; close the window or press ESC.')
        while not state['quit'] and p.isConnected(sim.client):
            events = poll_keys(sim)
            if pressed(events, KEY_ESC):
                break
            try:
                sim.step()
            except p.error:
                break
    finally:
        sim.close()


if __name__ == '__main__':
    main()
