"""实验 4.3 覆盖统计 v2：按阶段窗口 + 指标分布（用 records 的严格判据指标 + FK 阶段划分）。

A90 阶段（由动作夹爪 + FK 的 EE 高度划分）：
  approach（夹爪张开、闭合之前）→ grasp_close（动作夹爪首次 <0.03 到闭合后 5 帧）→
  lift（闭合后 EE z 上升 ≥3 cm）→ transport（抬升后水平位移 ≥3 cm 且夹爪闭合）→
  place_release（动作夹爪重新 >0.03 之后）→ settle（其余收尾帧）
B90 阶段：
  approach（EE x 在方块 x−1 cm 之前）→ push（严格判据给出的 robot_block 接触帧数）→
  wall_hold（wall 接触帧数）→ remaining
另统计：静止帧比例、初始 EE-物体距离、指标分布（lift/transport/advance/gap/force 等）。
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SRV = Path('/scratch/lijh/exp43')
MAN = Path('/scratch/lijh/exp40/repo/results/experiment_4_3/data_expansion/'
           'train_manifest_v2.json')
OUT = Path('/scratch/lijh/exp40/repo/results/experiment_4_3/data_expansion/coverage_v2.json')
EE_LINK = 11
STEPS_PER_FRAME = 12


def fk_series(qs):
    """对一串 7 维关节角做 FK，返回 EE 位置 (N,3)。"""
    import pybullet as p

    from exp41 import scene41

    scene = scene41.Scene41('B90', obj_pos=(0.55, 0.21, 0.0201), obj_yaw=np.pi / 2, robot_q=None)
    sim = scene.sim
    out = []
    try:
        for q in qs:
            for j in range(7):
                p.resetJointState(sim.robot, j, float(q[j]), targetVelocity=0.0,
                                  physicsClientId=sim.client)
            link = p.getLinkState(sim.robot, EE_LINK, computeForwardKinematics=True,
                                  physicsClientId=sim.client)
            out.append(link[4])
    finally:
        scene.close()
    return np.asarray(out, dtype=float)


def main() -> int:
    man = json.loads(MAN.read_text(encoding='utf-8'))
    eps = man['episodes']
    REC_METRICS = {}
    for tag in ('a90a', 'a90b', 'b90a', 'b90b'):
        for f in (SRV / tag / 'records').glob('*.json'):
            r = json.loads(f.read_text(encoding='utf-8'))
            REC_METRICS[(tag, r.get('seed'))] = r.get('metrics') or {}
    per_cfg = defaultdict(lambda: {'n': 0, 'frames': [], 'still_frac': [],
                                    'phases': defaultdict(list), 'metrics': defaultdict(list),
                                    'initial_ee_obj': []})
    rows = []
    for e in eps:
        import pandas as pd

        df = pd.read_parquet(SRV / e['parquet'])
        st = np.stack(df['observation.state'].to_numpy()).astype(float)
        ac = np.stack(df['action'].to_numpy()).astype(float)
        qs = st[:, :7]
        ee = fk_series(qs)
        dq = np.abs(ac[:, :7] - st[:, :7]).max(axis=1)
        gcmd = np.clip(ac[:, 7] / 0.04, 0, 1)
        n = len(df)
        rec = {'skill': e['skill'], 'gid': e['global_episode_id'], 'config': e['config_id'],
               'n_frames': n,
               'still_frac': float((dq < 1e-4).mean()),
               'moving_frac': float((dq > 0.01).mean())}
        start_obj = None
        # 初始 EE-物体距离（物体位姿取 records 里的 start_object）
        try:
            recs = json.loads((SRV / e['source_tag'] / 'records' /
                               f"{e.get('attempt_id') or ''}.json").read_text()) \
                if e.get('attempt_id') else None
        except Exception:  # noqa: BLE001
            recs = None
        phases = {}
        if e['skill'] == 'A90':
            # 用夹爪命令事件划分（稳健）：张开→闭合→（抓取+抬升+搬运）→张开（放置/释放）
            close_idx = np.where(gcmd < 0.6)[0]
            close_i = int(close_idx[0]) if len(close_idx) else n
            reap = np.where(gcmd > 0.85)[0]
            reap = reap[reap > close_i]
            reopen_i = int(reap[0]) if len(reap) else n
            phases = {
                'approach': close_i,
                'grasp_lift_transport': max(0, reopen_i - close_i),
                'place_release': max(0, n - reopen_i),
            }
        else:
            m = dict(e.get('metrics') or {})
            rec_metrics = REC_METRICS.get((e['source_tag'], e['seed']), {})
            for k in ('robot_block_contact_steps', 'wall_contact_steps'):
                if m.get(k) is None and rec_metrics.get(k) is not None:
                    m[k] = rec_metrics[k]
            push_frames = int(round((m.get('robot_block_contact_steps') or 0) / STEPS_PER_FRAME))
            wall_frames = int(round((m.get('wall_contact_steps') or 0) / STEPS_PER_FRAME))
            approach = max(0, n - push_frames - wall_frames)
            phases = {'approach': approach, 'push': min(push_frames, n),
                      'wall_hold': min(wall_frames, n),
                      'remaining': max(0, n - approach - push_frames - wall_frames)}
        rec['phases'] = phases
        rec['metrics'] = {k: (e.get('metrics') or {}).get(k) for k in
                          ('lift_m', 'transport_m', 'position_error_m', 'two_finger_overlap_steps',
                           'slip_after_release_m', 'advance_x_m', 'lateral_drift_m',
                           'wall_contact_steps', 'wall_contact_force_max_n', 'terminal_gap_m',
                           'robot_block_contact_steps')}
        rows.append(rec)
        key = f"{e['skill']}:{e['config_id']}"
        c = per_cfg[key]
        c['n'] += 1
        c['frames'].append(n)
        c['still_frac'].append(rec['still_frac'])
        for k, v in phases.items():
            c['phases'][k].append(v)
        for k, v in rec['metrics'].items():
            if v is not None:
                c['metrics'][k].append(float(v))

    def summ(vals):
        if not vals:
            return None
        a = np.asarray(vals, dtype=float)
        return {'mean': float(a.mean()), 'median': float(np.median(a)), 'min': float(a.min()),
                'max': float(a.max())}

    out = {'n_episodes': len(rows), 'by_config': {}, 'by_skill': {}, 'rows': rows}
    for skill in ('A90', 'B90'):
        sel = [r for r in rows if r['skill'] == skill]
        phase_tot = defaultdict(float)
        for r in sel:
            for k, v in r['phases'].items():
                phase_tot[k] += v
        out['by_skill'][skill] = {
            'n': len(sel),
            'frames': summ([r['n_frames'] for r in sel]),
            'still_frac': summ([r['still_frac'] for r in sel]),
            'moving_frac': summ([r['moving_frac'] for r in sel]),
            'phase_frames_total': {k: int(v) for k, v in phase_tot.items()},
            'phase_frac_total': {k: float(v / max(sum(phase_tot.values()), 1))
                                  for k, v in phase_tot.items()},
            'metrics': {k: summ([r['metrics'].get(k) for r in sel
                                 if r['metrics'].get(k) is not None])
                        for k in ('lift_m', 'transport_m', 'position_error_m',
                                  'advance_x_m', 'lateral_drift_m', 'wall_contact_steps',
                                  'wall_contact_force_max_n', 'terminal_gap_m',
                                  'robot_block_contact_steps')},
        }
    for key, c in sorted(per_cfg.items()):
        out['by_config'][key] = {
            'n': c['n'],
            'frames': summ(c['frames']),
            'still_frac': summ(c['still_frac']),
            'phase_frames_mean': {k: float(np.mean(v)) for k, v in c['phases'].items()},
            'metrics': {k: summ(v) for k, v in c['metrics'].items()},
        }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding='utf-8')
    print(json.dumps({s: {'n': out['by_skill'][s]['n'],
                          'frames': out['by_skill'][s]['frames'],
                          'still_frac': out['by_skill'][s]['still_frac'],
                          'phase_frac': out['by_skill'][s]['phase_frac_total']}
                      for s in ('A90', 'B90')}, ensure_ascii=False, indent=2))
    print('wrote', OUT)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
