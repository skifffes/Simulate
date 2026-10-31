import csv
from pathlib import Path
B=Path(r'D:\python\simulate\results\experiment_2_11');D=B/'data'
audit=[
 {'source_experiment':'2.6_B_gripper_variation','state_id':'ep66_f39','episode_id':66,'gripper_mm':14,'robot_q_available':True,'object_pose_available':True,'top_rgb_available':True,'wrist_rgb_available':True,'expert_action_available':True,'timestamps_available':True,'physics_trace_available':True,'first_contact_available':True,'usable_for_teacher_forcing':True,'limitations':'RGB captured by replay; original run had video only'},
 {'source_experiment':'2.6_B_gripper_variation','state_id':'ep66_f39','episode_id':66,'gripper_mm':35,'robot_q_available':True,'object_pose_available':True,'top_rgb_available':True,'wrist_rgb_available':True,'expert_action_available':True,'timestamps_available':True,'physics_trace_available':True,'first_contact_available':True,'usable_for_teacher_forcing':True,'limitations':'RGB captured by replay'},
 {'source_experiment':'2.11_C3_expert','state_id':'C3_seed40000000','episode_id':None,'gripper_mm':14,'robot_q_available':True,'object_pose_available':True,'top_rgb_available':True,'wrist_rgb_available':True,'expert_action_available':True,'timestamps_available':True,'physics_trace_available':True,'first_contact_available':True,'usable_for_teacher_forcing':True,'limitations':'expert newly run; RGB captured by replay'},
 {'source_experiment':'2.11_C3_expert','state_id':'C3_seed40000000','episode_id':None,'gripper_mm':35,'robot_q_available':True,'object_pose_available':True,'top_rgb_available':True,'wrist_rgb_available':True,'expert_action_available':True,'timestamps_available':True,'physics_trace_available':True,'first_contact_available':True,'usable_for_teacher_forcing':True,'limitations':'expert newly run; RGB captured by replay'},
 {'source_experiment':'2.11_C3_expert','state_id':'C3_seed40000000','episode_id':None,'gripper_mm':40,'robot_q_available':True,'object_pose_available':True,'top_rgb_available':True,'wrist_rgb_available':True,'expert_action_available':True,'timestamps_available':True,'physics_trace_available':True,'first_contact_available':True,'usable_for_teacher_forcing':True,'limitations':'40mm is out of B90 training range'},
]
with (D/'existing_expert_data_audit.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(audit[0].keys()));w.writeheader();w.writerows(audit)
paired=[
 {'state_id':'ep66_f39','source':'B90 training frame','widths_mm':'14;35','same_q':True,'same_object':True,'note':'real B90 training midstate; 35 has no local training support'},
 {'state_id':'C3_seed40000000','source':'A90+C3 connector terminal','widths_mm':'14;35;40','same_q':True,'same_object':True,'note':'35/40 are out-of-local-coverage; 40 out-of-global B90 range'},
]
with (D/'paired_initial_states.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(paired[0].keys()));w.writeheader();w.writerows(paired)
ident=[
 {'question':'Q1 available data','status':'Observed','note':'5 expert trajectories with q/action/video; RGB replayed; no contact labels in parquet'},
 {'question':'Q2 expert 14 vs35 action difference','status':'Observed','note':'ep66/C3 expert initial target nearly identical because expert closes to14 before approach; any later difference is trajectory divergence'},
 {'question':'Q3 C3 expert 14/35/40 approach','status':'Observed','note':'expert follows same fixed 0.07->0.018 path after closing; width does not change waypoint'},
 {'question':'Q4 D2 joint target error','status':'Observed','note':'q L2 ~0.10-0.20 rad at h0; EE z error 1-18mm'},
 {'question':'Q5 model vs expert conditional response','status':'Supported','note':'model 35-14 higher than expert at h0/h19; C3 40 even larger'},
 {'question':'Q6 C3 deviation mode','status':'Supported','note':'higher initial target and delayed descent for 35/40; 40 strongest'},
 {'question':'Q7 link to 2.10 gap','status':'Descriptive only','note':'both C3/ep66 wide local support is zero; no causal identification'},
 {'question':'Q8 training intervention hypothesis','status':'Candidate only','note':'fixed-q wide 35 expert trajectories at C3/ep66; 40 separate extrapolation'},
]
with (D/'identifiability_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(ident[0].keys()));w.writeheader();w.writerows(ident)
print('ok')
