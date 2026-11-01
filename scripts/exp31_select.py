import csv,json,numpy as np
from pathlib import Path
R=Path(r'D:\python\simulate\results\experiment_3_2');rows=list(csv.DictReader((R/'b90_all_candidates.csv').open(encoding='utf-8-sig')))
sel_ids=['ep079_f000','ep104_f000','ep082_f000','ep080_f000','ep074_f000','ep068_f000'];anchor='ep064_f000'
# thresholds from full 60 pre-rollout
vals=np.array([float(x['obj_pos_diff_mm']) for x in rows]);near,medium=np.quantile(vals,[1/3,2/3])
for x in rows:
 v=float(x['obj_pos_diff_mm']);x['scene_stratum']='Near' if v<=near else ('Medium' if v<=medium else 'Far')
sel=[x for x in rows if x['b_candidate_id'] in sel_ids]
anchor_row=next(x for x in rows if x['b_candidate_id']==anchor)
obj={'source':'results/experiment_3_2/b90_all_candidates.csv','selection_rule':'pre-registered: 2 Near +2 Medium +2 Far by object-position difference; anchor ep064; no D2 success used','near_threshold_mm':float(near),'medium_threshold_mm':float(medium),'targets':[x['b_candidate_id'] for x in sel]+[anchor],'anchor':anchor,'unique_q_count':len(set(round(float(x['q_dist_rad']),6) for x in rows)),'note':'ep064 is anchor, not counted in the six new targets'}
(R/'candidate_selection.json').write_text(json.dumps(obj,indent=2))
# scene compatibility summary
summary=[]
for s in ('Near','Medium','Far'):
 ss=[x for x in rows if x['scene_stratum']==s];summary.append({'stratum':s,'n':len(ss),'obj_min_mm':min(float(x['obj_pos_diff_mm']) for x in ss),'obj_max_mm':max(float(x['obj_pos_diff_mm']) for x in ss),'goal_valid':sum(x['goal_valid_A']=='True' for x in ss),'path_valid':sum(x['path_valid_A']=='True' for x in ss)})
with (R/'scene_compatibility.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(summary[0].keys()));w.writeheader();w.writerows(summary)
# joint coverage: q classes from modes
cov=[]
for mode in sorted(set(x['start_mode'] for x in rows)):
 ss=[x for x in rows if x['start_mode']==mode];cov.append({'start_mode':mode,'n_episodes':len(ss),'unique_q':len(set(round(float(x['q_dist_rad']),6) for x in ss)),'near':sum(x['scene_stratum']=='Near' for x in ss),'medium':sum(x['scene_stratum']=='Medium' for x in ss),'far':sum(x['scene_stratum']=='Far' for x in ss)})
with (R/'joint_state_coverage.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(cov[0].keys()));w.writeheader();w.writerows(cov)
# selected table
with (R/'selected_targets.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(sel[0].keys())+['role']);w.writeheader();w.writerows([{**x,'role':'anchor' if x['b_candidate_id']==anchor else 'new'} for x in sel+[anchor_row]])
print('near',near,'medium',medium,'selected',len(sel))
