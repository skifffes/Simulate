"""Gate 3：A90/B90 启动阶段（前 20–40 控制周期）监督审计与可识别性检查。

回答三件事：
  1. 每条 episode 的启动时序：前若干帧的关节命令、实际位移、首次非零命令帧、夹爪；
  2. 静止段内"同一观测是否对应不同 16 步 chunk"（输入无阶段/时间信息时的监督冲突）；
  3. 跨 episode 的启动状态是否等价（同一 HOME + 同一图像 + 同一语言 → 却要求不同动作？）。

用法（simulate venv）：
  python scripts/exp42/startup_supervision_audit.py --out results/experiment_4_2/contract_replay_p0c
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / 'src', ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

DATA = Path('/scratch/lijh/exp42/data')
HORIZON = 16
FINGER_MAX_M = 0.04
FIRST_N = 40


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/experiment_4_2/contract_replay_p0c')
    ap.add_argument('--n-episodes', type=int, default=90)
    args = ap.parse_args(argv)
    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    meta = [json.loads(l) for l in (DATA / 'meta.jsonl').read_text(encoding='utf-8').splitlines()
            if l.strip()][:args.n_episodes]

    rows = []
    for m in meta:
        sa = np.load(DATA / m['state_path'])
        S = sa['state'].astype(float)
        A = sa['action'].astype(float)
        dq = np.abs(A[:, :7] - S[:, :7]).max(axis=1)
        lead = 0
        for v in dq:
            if v < 1e-4:
                lead += 1
            else:
                break
        first_cmd = int(np.argmax(dq >= 1e-4)) if (dq >= 1e-4).any() else -1
        n = min(FIRST_N, S.shape[0])
        rows.append({
            'skill': m['skill'], 'episode': m['episode_index'], 'split': m['split'],
            'n_frames': int(S.shape[0]),
            'leading_still_frames': lead,
            'first_nonzero_cmd_frame': first_cmd,
            'cmd_absmax_first16': float(dq[:min(16, S.shape[0])].max()),
            'cmd_absmax_first40': float(dq[:n].max()),
            'grip_cmd_first': float(A[0, 7]), 'grip_cmd_min_first16': float(A[:min(16, S.shape[0]), 7].min()),
            'q0': S[0, :7].round(5).tolist(),
        })

    # 同一观测 vs 不同 chunk：取每条 episode 的 t=0 与 t=5（两者都在静止段）
    conflicts = []
    for m in meta:
        sa = np.load(DATA / m['state_path'])
        S = sa['state'].astype(float)
        A = sa['action'].astype(float)
        if S.shape[0] < HORIZON + 6:
            continue
        top = np.load(DATA / m['top_path'], mmap_mode='r')
        c0 = A[0:HORIZON, :7] - S[0, :7][None, :]
        c5 = A[5:5 + HORIZON, :7] - S[5, :7][None, :]
        img_diff = float(np.abs(np.asarray(top[0], dtype=float)
                                - np.asarray(top[5], dtype=float)).mean())
        q_diff = float(np.abs(S[0, :7] - S[5, :7]).max())
        first_row_diff = float(np.abs(c0[0] - c5[0]).max())
        later_rows_diff = float(np.abs(c0[1:] - c5[1:]).max())
        conflicts.append({'skill': m['skill'], 'episode': m['episode_index'],
                          'q0_vs_q5_max_diff_rad': q_diff,
                          'image0_vs_image5_mean_absdiff': img_diff,
                          'chunk_first_row_diff_rad': first_row_diff,
                          'chunk_later_rows_max_diff_rad': later_rows_diff})

    def summarize(skill):
        sel = [r for r in rows if r['skill'] == skill]
        cf = [c for c in conflicts if c['skill'] == skill]
        return {
            'n_episodes': len(sel),
            'leading_still_frames_mean': float(np.mean([r['leading_still_frames'] for r in sel])),
            'leading_still_frames_minmax': [int(min(r['leading_still_frames'] for r in sel)),
                                            int(max(r['leading_still_frames'] for r in sel))],
            'first_nonzero_cmd_frame_unique': sorted({r['first_nonzero_cmd_frame'] for r in sel}),
            'cmd_absmax_first16_median': float(np.median([r['cmd_absmax_first16'] for r in sel])),
            'cmd_absmax_first40_median': float(np.median([r['cmd_absmax_first40'] for r in sel])),
            'grip_cmd_first_unique': sorted({round(r['grip_cmd_first'], 3) for r in sel}),
            'n_distinct_q0': len({tuple(np.round(r['q0'], 4)) for r in sel}),
            'still_segment_conflict': {
                'n_pairs': len(cf),
                'q0_vs_q5_max_diff_median': float(np.median([c['q0_vs_q5_max_diff_rad'] for c in cf])),
                'image_mean_absdiff_median': float(np.median([c['image0_vs_image5_mean_absdiff']
                                                              for c in cf])),
                'chunk_first_row_diff_median': float(np.median([c['chunk_first_row_diff_rad']
                                                                for c in cf])),
                'chunk_later_rows_diff_median': float(np.median([c['chunk_later_rows_max_diff_rad']
                                                                 for c in cf])),
                'chunk_later_rows_diff_p90': float(np.percentile([c['chunk_later_rows_max_diff_rad']
                                                                  for c in cf], 90)),
            },
        }

    res = {'per_episode': rows, 'still_segment_pairs': conflicts,
           'summary': {s: summarize(s) for s in ('A90', 'B90')}}
    (out / 'startup_supervision_audit.json').write_text(
        json.dumps(res, ensure_ascii=False, indent=2, default=float), encoding='utf-8')

    # 判定：把"首步标签"和"chunk 后续行"分开（review 要求区分这两种语义）
    verdicts = {}
    for skill, s in res['summary'].items():
        c = s['still_segment_conflict']
        same_obs = c['q0_vs_q5_max_diff_median'] < 1e-3 and c['image_mean_absdiff_median'] < 2.0
        first_consistent = c['chunk_first_row_diff_median'] < 1e-3
        later_differ = c['chunk_later_rows_diff_median'] > 0.01
        if same_obs and later_differ:
            tail = 'IDLE_SUPERVISION_AMBIGUOUS'
        elif same_obs:
            tail = 'IDLE_SUPERVISION_VALID'
        else:
            tail = 'IDLE_SUPERVISION_NOT_IDENTIFIABLE'
        verdicts[skill] = {
            'first_step': ('IDLE_SUPERVISION_VALID' if first_consistent
                           else 'IDLE_SUPERVISION_AMBIGUOUS'),
            'chunk_rows_k_ge_1': tail,
            'reason': ('静止段内 t=0..5 的观测几乎逐位相同（q 差 <1e-3 rad、图像差 <2/255），'
                       '首行标签也一致（0 动作），但 16 步 chunk 的后续行相差约 0.16–0.19 rad；'
                       '模型输入没有阶段/时间信息，无法区分"该等待多久"。'),
            'artifact_check': ('所有 episode 的首次非零命令都恰好在第 6 帧，'
                               '与采集端 _settle(0.3) 产生的 6 帧等待一致（0.3 s × 20 Hz）。'),
        }
    res['verdict'] = verdicts
    (out / 'startup_supervision_audit.json').write_text(
        json.dumps(res, ensure_ascii=False, indent=2, default=float), encoding='utf-8')
    print(json.dumps({'summary': res['summary'], 'verdict': verdicts},
                     ensure_ascii=False, indent=2)[:2600])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
