"""CPU-only contract counterexamples; does not run a policy or change deployment."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

from exp40.action_adapter import ActionAdapterConfig, adapt_action, chunk_to_targets
from scripts.exp42.train_lora import HORIZON, make_item, windows


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=ROOT / 'results/experiment_4_2/independent_audit')
    ap.add_argument('--data', type=Path, help='Optional existing EXP42_DATA cache; read only')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    q0 = np.array([0.0, -0.4, 0.0, -1.8, 0.0, 1.4, 0.7])
    state = np.tile(np.r_[q0, .04, .04], (HORIZON, 1))
    command = np.tile(np.r_[q0, .017], (HORIZON, 1))
    command[:, 0] += .01 * np.arange(1, HORIZON + 1)
    ep = {'state': state, 'action': command,
          'top': np.zeros((HORIZON, 2, 2, 3), dtype=np.uint8),
          'wrist': np.zeros((HORIZON, 2, 2, 3), dtype=np.uint8)}
    chunk = make_item({'prompt': 'synthetic contract fixture'}, ep, 0)['actions']
    lo, hi = np.full(7, -3.0), np.full(7, 3.0)
    cfg = ActionAdapterConfig(gripper_mode='continuous')
    legacy = chunk_to_targets(chunk, q0, lo, hi, cfg)
    candidate = [adapt_action(row, q0, lo, hi, cfg) for row in chunk]
    old_q = np.stack([a.q_target for a in legacy])
    new_q = np.stack([a.q_target for a in candidate])
    assert all(a.accepted and not a.clipped_dims for a in legacy + candidate)
    np.testing.assert_allclose(old_q[0], command[0, :7], atol=1e-12)
    assert not np.allclose(old_q[1:], command[1:, :7])
    np.testing.assert_allclose(new_q, command[:, :7], atol=1e-12)

    tree = ast.parse((ROOT / 'scripts/exp42/eval_rollouts.py').read_text(encoding='utf-8'))
    loops = [n for n in ast.walk(tree) if isinstance(n, ast.For)
             and any(isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                     and c.func.attr == 'setJointMotorControl2' for c in ast.walk(n))]
    motor_loop = min(loops, key=lambda n: n.end_lineno - n.lineno)
    sent = []
    fake_p = SimpleNamespace(POSITION_CONTROL=2,
                            setJointMotorControl2=lambda *a, **kw: sent.append(float(a[3])))
    scope = {'p': fake_p, 'scene': SimpleNamespace(sim=SimpleNamespace(robot=1, client=0)),
             'adapted': candidate, 'i': 0, 'float': float}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[motor_loop], type_ignores=[])),
                 'eval_rollouts_motor_loop', 'exec'), scope)
    assert len(sent) == 2
    assert all(abs(v - .425) < 1e-12 for v in sent)
    assert all(abs(v - .017) > .4 for v in sent)
    assert abs(candidate[0].gripper_target_m - .017) < 1e-12
    binary_width = adapt_action(chunk[0], q0, lo, hi).gripper_target_m
    assert binary_width == 0.0

    with (args.out / 'synthetic_contract_trace.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['provenance', 'chunk_index', 'expert_target_joint0_rad',
                         'label_joint0_rad', 'legacy_target_joint0_rad',
                         'anchor_candidate_joint0_rad', 'legacy_gripper_command_m',
                         'continuous_candidate_gripper_m'])
        for k in range(HORIZON):
            writer.writerow(['SYNTHETIC_NOT_MODEL', k, command[k, 0], chunk[k, 0],
                             old_q[k, 0], new_q[k, 0], sent[0], candidate[k].gripper_target_m])

    historical = json.loads((ROOT / 'results/experiment_4_2/still_frames.json').read_text(encoding='utf-8'))
    rows = historical['episodes']
    counts = {}
    for skill in ('A90', 'B90'):
        subset = [r for r in rows if r['skill'] == skill]
        by_split = {}
        for split in ('train', 'val', 'test'):
            selected = [r for r in subset if r['split'] == split]
            by_split[split] = {'episodes': len(selected), 'frames': sum(r['n_frames'] for r in selected),
                               'windows': sum(len(windows([r], split)) for r in selected)}
        counts[skill] = {'source': 'historical still_frames.json, NOT raw-data rescan',
                         'leading_still_histogram': dict(Counter(r['leading_still_frames'] for r in subset)),
                         'splits': by_split}
    train_windows = sum(v['splits']['train']['windows'] for v in counts.values())
    raw_status = 'NOT_EXECUTED: no local raw EXP42_DATA supplied'
    if args.data:
        meta_path = args.data / 'meta.jsonl'
        meta = [json.loads(s) for s in meta_path.read_text(encoding='utf-8').splitlines() if s.strip()]
        with (args.out / 'raw_first40.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['skill', 'episode', 'frame', 'q_rad', 'target_rad', 'next_q_rad',
                             'target_minus_q_rad', 'actual_delta_rad', 'tracking_residual_rad',
                             'gripper_target_m', 'sa_sha256'])
            for skill in ('A90', 'B90'):
                selected = sorted((m for m in meta if m['skill'] == skill), key=lambda m: m['episode_index'])[:3]
                if len(selected) != 3:
                    raise ValueError(f'{skill}: need at least 3 episodes')
                for row in selected:
                    path = args.data / row['state_path']
                    with np.load(path, allow_pickle=False) as sa:
                        s, a = sa['state'], sa['action']
                    digest = sha256(path)
                    for t in range(min(40, len(s) - 1)):
                        vectors = [s[t, :7], a[t, :7], s[t+1, :7], a[t, :7]-s[t, :7],
                                   s[t+1, :7]-s[t, :7], a[t, :7]-s[t+1, :7]]
                        writer.writerow([skill, row['episode_index'], t,
                                         *[json.dumps(v.tolist()) for v in vectors], float(a[t, 7]), digest])
        raw_status = 'EXECUTED: six episodes first40; not RGB synchronization validation'
    sources = ['scripts/exp42/train_lora.py', 'scripts/exp42/eval_rollouts.py',
               'src/exp40/action_adapter.py', 'src/vla/atomic_tasks/recording.py',
               'results/experiment_4_2/still_frames.json',
               'results/experiment_4_2/checkpoint_manifest.json']
    try:
        head = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        head_file = ROOT / '.git/HEAD'
        if head_file.is_file():
            ref = head_file.read_text().strip()
            ref_file = ROOT / '.git' / ref.removeprefix('ref: ')
            head = ref_file.read_text().strip() if ref.startswith('ref: ') and ref_file.is_file() else ref
        else:
            head = 'UNAVAILABLE_NON_GIT_COPY_USE_SOURCE_SHA256'
    result = {'status': 'PARTIAL', 'head': head, 'fixture': 'SYNTHETIC_NOT_MODEL_NOT_PHYSICS',
              'checks': {'legacy_chunk_expected_failure_reproduced': True,
                         'anchor_candidate_roundtrip_pass': True,
                         'legacy_gripper_units_expected_failure_reproduced': True,
                         'continuous_candidate_units_pass': True,
                         'binary_candidate_still_mismatches_expert': True},
              'legacy_chunk_max_target_error_rad': float(np.max(np.abs(old_q-command[:, :7]))),
              'anchor_candidate_max_target_error_rad': float(np.max(np.abs(new_q-command[:, :7]))),
              'gripper_command_ratio': sent[0] / .017,
              'counts_from_historical_metadata': counts,
              'nominal_901x4_window_draws_per_train_window': 901 * 4 / train_windows,
              'raw_frames': raw_status,
              'not_executed': ['checkpoint multisampling', 'model tensor hash comparison',
                               'real FM loss', 'EE/Jacobian/physics', 'remote disk inspection'],
              'source_sha256': {s: sha256(ROOT / s) for s in sources}}
    (args.out / 'local_test_results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    behavior_path = args.out / 'model_behavior_diagnostics.csv'
    if not behavior_path.exists():
        with behavior_path.open('w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(['status', 'skill', 'phase', 'model', 'seed', 'predicted_actions',
                                   'expert_actions', 'cosine7', 'action_norm', 'joint_mae',
                                   'ee_direction', 'prediction_variance', 'gripper_error'])
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
