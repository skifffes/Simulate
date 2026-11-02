"""分析最终 B90 采集的 17 次失败：失败原因与终止间隙分布（RQ6 用）。"""
import collections
import glob
import json

rows = []
for f in sorted(glob.glob('/scratch/lijh/exp40/repo/results/experiment_4_1/dataset/records/*.json')):
    r = json.load(open(f, encoding='utf-8'))
    if r['skill'] != 'B90':
        continue
    rows.append(r)

fails = [r for r in rows if not r['strict_success']]
print('n_b90', len(rows), 'fails', len(fails))
print('failure reasons:', collections.Counter(r['failure_reason'] for r in fails))
print('expert reasons among fails:', collections.Counter(r['expert_reason'] for r in fails))
gaps = []
for r in fails:
    g = r['metrics'].get('terminal_gap_m')
    if g is not None:
        gaps.append(round(float(g), 4))
print('terminal_gap of failures (m):', sorted(gaps))
print('gaps <= 0.02:', sum(1 for g in gaps if abs(g) <= 0.02), 'of', len(gaps))
# 这些 gap<=2cm 的失败：旧判据的其他条件是否满足？
for r in fails:
    m = r['metrics']
    g = m.get('terminal_gap_m')
    if g is None or abs(g) > 0.02:
        continue
    print(' near-wall failure:', r['attempt_id'], 'gap', round(g, 4), 'dy', round(m['lateral_drift_m'], 4),
          'yaw_deg', round(m['yaw_change_rad'] * 57.2958, 2), 'tilt', round(m['tilt_deg'], 2),
          'speed', m['final_speed_m_s'], 'reason', r['failure_reason'])
succ = [r for r in rows if r['strict_success']]
print('success: n', len(succ),
      'wall_contact_steps min/max', min(r['metrics']['wall_contact_steps'] for r in succ),
      max(r['metrics']['wall_contact_steps'] for r in succ),
      'force max min', round(min(r['metrics']['wall_contact_force_max_n'] for r in succ), 3))
print('success terminal gaps:', sorted(round(r['metrics']['terminal_gap_m'], 4) for r in succ)[:5], '...',
      sorted(round(r['metrics']['terminal_gap_m'], 4) for r in succ)[-5:])
print('success advance min:', round(min(r['metrics']['advance_x_m'] for r in succ), 4))
print('success tilt max:', round(max(r['metrics']['tilt_deg'] for r in succ), 3),
      'speed max:', max(r['metrics']['final_speed_m_s'] for r in succ))
# A90 统计
a = [json.load(open(f, encoding='utf-8')) for f in sorted(glob.glob('/scratch/lijh/exp40/repo/results/experiment_4_1/dataset/records/*.json'))]
a = [r for r in a if r['skill'] == 'A90']
print('\nA90 n', len(a), 'success', sum(1 for r in a if r['strict_success']))
print('A90 pos err max', round(max(r['metrics']['position_error_m'] for r in a), 4),
      'yaw deg max', round(max(r['metrics']['yaw_error_deg'] for r in a), 2),
      'lift min', round(min(r['metrics']['lift_m'] for r in a), 4),
      'transport min', round(min(r['metrics']['transport_m'] for r in a), 4))
