import ast, csv, json
from pathlib import Path

ROOT = Path(r'D:\python\simulate')
B = ROOT / 'results/experiment_2_6'
for d in ['A_midstate_launch/expert_reference', 'A_midstate_launch/frozen_D2',
          'A_midstate_launch/comparisons', 'B_gripper_variation/expert',
          'B_gripper_variation/frozen_D2', 'B_gripper_variation/comparisons',
          'C3_reference', 'videos']:
    (B / d).mkdir(parents=True, exist_ok=True)

recon = json.loads((B / 'A_midstate_launch/reconstruction_report.json').read_text(encoding='utf-8'))
with (B / 'A_midstate_launch/state_manifest.csv').open('w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f); w.writerow(['state', 'episode', 'frame', 'group', 'mode', 'seed',
                                   'q_error_rad', 'gripper_error_m', 'obj_x', 'obj_y', 'obj_z',
                                   'bullet'])
    for r in recon:
        w.writerow([f"ep{r['episode']:03d}_f{r['frame']:03d}", r['episode'], r['frame'],
                    r['group'], r['mode'], r['seed'], r['q_error_rad'], r['gripper_error_m'],
                    r['obj_xy'][0], r['obj_xy'][1], r['obj_z'], r['bullet']])
with (B / 'A_midstate_launch/reconstruction_audit.csv').open('w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f); w.writerow(['state', 'episode', 'frame', 'q_error_rad', 'gripper_error_m',
                                   'status'])
    for r in recon:
        status = 'valid' if r['q_error_rad'] < 1e-4 and r['gripper_error_m'] < 1e-4 else 'reconstruction_failed'
        w.writerow([f"ep{r['episode']:03d}_f{r['frame']:03d}", r['episode'], r['frame'],
                    r['q_error_rad'], r['gripper_error_m'], status])

rows = list(csv.DictReader((B / 'summary.csv').open(encoding='utf-8-sig')))
with (B / 'B_gripper_variation/validity_checks.csv').open('w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f); w.writerow(['state', 'requested_m', 'actual_m', 'obj_move_mm',
                                   'tilt_change_deg', 'valid'])
    for r in rows:
        if not r['gripper_init']:
            continue
        g = ast.literal_eval(r['gripper_init'])
        valid = g['obj_move_mm'] < 2.0 and abs(g['obj_tilt_change_deg']) < 2.0
        w.writerow([r['state'], g['requested_m'], g['actual_m'], g['obj_move_mm'],
                    g['obj_tilt_change_deg'], valid])
with (B / 'B_gripper_variation/candidate_states.csv').open('w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f); w.writerow(['state', 'group', 'mode', 'episode', 'frame',
                                   'expert_success', 'frozen_success', 'max_lift_m',
                                   'wall_contact', 'final_tilt_deg'])
    for r in rows:
        if r['gripper_init']:
            w.writerow([r['state'], r['group'], r['mode'], r['source_episode'],
                        r['source_frame'], r['condition'] == 'expert' and r['success'],
                        r['condition'] == 'frozen' and r['success'], r['max_lift_m'],
                        r['wall_contact'], r['final_tilt_deg']])
# video manifest
with (B / 'video_manifest.csv').open('w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f); w.writerow(['condition', 'state', 'success', 'top', 'side', 'contact',
                                   'comparison', 'events'])
    for r in rows:
        base = ('A_midstate_launch' if not r['gripper_init'] else 'B_gripper_variation')
        sub = r['condition']
        d = f'/scratch/lijh/simulate_vla/results/experiment_2_6/{base}/{sub}/{r["state"]}'
        w.writerow([r['condition'], r['state'], r['success'],
                    d + '/top.mp4', d + '/side.mp4', d + '/contact.mp4',
                    d + '/comparison.mp4', d + '/events.mp4'])
print('done')
