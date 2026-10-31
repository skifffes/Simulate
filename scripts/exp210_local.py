import csv,json
from pathlib import Path
import numpy as np
ROOT=Path(r'D:\python\simulate');B=ROOT/'results/experiment_2_10';D=B/'data'
# dataset audit
audit=[
 {'dataset':'lerobot_panda_e23_D2','task':'A0_yaw_longY','episodes':120,'frames':16164,'language':'pick up ... long side aligned with Y'},
 {'dataset':'lerobot_panda_e23_D2','task':'A90_yaw_longX','episodes':120,'frames':21450,'language':'pick up ... long side aligned with X'},
 {'dataset':'lerobot_panda_e23_D2','task':'B90_push_yaw90','episodes':60,'frames':7016,'language':'push ... positive X'},
 {'dataset':'lerobot_panda_e23_D2','task':'B0_push_yaw0','episodes':60,'frames':7637,'language':'push ... positive X'},
 {'dataset':'lerobot_panda_e23_D2','task':'C_tray','episodes':120,'frames':16625,'language':'pick up ... into tray'},
 {'dataset':'lerobot_panda_e23_D2','task':'TOTAL','episodes':480,'frames':68892,'language':''},
]
with (D/'dataset_audit.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(audit[0].keys()));w.writeheader();w.writerows(audit)
# policy response: compute 14->35 and 14->40 from saved 2.8 chunks
ps=[]
for f in sorted((ROOT/'results/experiment_2_8/policy').glob('*.npz')):
 z=np.load(f)['z'];d35=(z[4,:,9]-z[0,:,9])*1000;d40=(z[5,:,9]-z[0,:,9])*1000
 ps.append({'state_id':f.stem,'d35':float(d35.mean()),'d40':float(d40.mean())})
cov={}
for r in csv.DictReader((D/'local_joint_coverage.csv').open(encoding='utf-8-sig')):
 if abs(float(r['radius_rad'])-0.75)<1e-6:cov[r['state_id']]=r
rows=[]
for x in ps:
 c=cov.get(x['state_id'],{})
 rows.append({'state_id':x['state_id'],'source':'2.8_B90_midstate','model_delta_14_35_step10_mm':x['d35'],
              'model_delta_14_40_step10_mm':x['d40'],'local_narrow_frames':c.get('narrow_frames',''),
              'local_wide_frames':c.get('wide_frames',''),'common_support':c.get('common_support','')})
rows.append({'state_id':'C3_seed40000000','source':'2.7_C3_counterfactual','model_delta_14_35_step10_mm':16.96,
             'model_delta_14_40_step10_mm':32.04,'local_narrow_frames':142,'local_wide_frames':0,'common_support':False})
with (D/'policy_response.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
# supported comparison rows
sup=[]
for r in csv.DictReader((D/'local_joint_coverage.csv').open(encoding='utf-8-sig')):
 if r['common_support']=='True':
  pr=next(x for x in rows if x['state_id']==r['state_id'])
  sup.append({'state_id':r['state_id'],'radius_rad':r['radius_rad'],'narrow_frames':r['narrow_frames'],'narrow_episodes':r['narrow_episodes'],
              'wide_frames':r['wide_frames'],'wide_episodes':r['wide_episodes'],
              'expert_dz_h0_narrow':r['expert_dz_h0_narrow'],'expert_dz_h0_wide':r['expert_dz_h0_wide'],
              'model_delta_14_35_step10_mm':pr['model_delta_14_35_step10_mm'],
              'comparison_valid':False,'reason':'wide frames only initial phase/high EE; narrow spans descent/push; no matched phase'})
with (D/'supported_policy_expert_comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(sup[0].keys()));w.writeheader();w.writerows(sup)
# identifiability
ident=[
 {'question':'Q1 gripper-action height relation','status':'Observed/Supported','evidence':'wide 60 frames only initial; narrow spans close/descend/push; global h0 weak','counter':'wide future h19 -167mm; not simply wide=high','confound':'phase/EE/q'},
 {'question':'Q2 fixed-q wide/narrow expert contrast','status':'Not Identifiable','evidence':'0/43 states common at 0.30; 13 rows @0.50/0.75 only early wide','counter':'','confound':'no matched phase/geometry'},
 {'question':'Q3 cause of non-identifiability','status':'Observed','evidence':'wide only 5-7 initial frames; no object frame labels; frames clustered','counter':'','confound':''},
 {'question':'Q4 C3 q coverage','status':'Supported (q-only)','evidence':'C3 126-147 frames/3-4 eps at0.75','counter':'0 wide frames','confound':'q-only, no object'},
 {'question':'Q5 ep66 f39 role','status':'Supported (14mm only)','evidence':'1731 q-neighbors/52 eps at0.75, all narrow','counter':'0 wide; cannot infer +35/+40','confound':''},
 {'question':'Q6 model vs expert in supported region','status':'Not Identifiable','evidence':'no valid wide/narrow phase-matched comparison','counter':'','confound':'coverage/phase'},
 {'question':'H1 expert timing association','status':'Supported/Weakened','evidence':'close cmd 0s, actual close 0.35s, descent 0.30s, low 2.0s; wide only initial','counter':'descent starts before actual close; wide future targets descend','confound':'hard-coded expert'},
 {'question':'H2 local joint supervision gap','status':'Observed; causal Not Identifiable','evidence':'C3 wide=0; ep66 wide=0','counter':'gap alone does not predict direction','confound':'no wide labels'},
 {'question':'H3 expert conditional action difference','status':'Not Identifiable','evidence':'no fixed-state wide/narrow labels','counter':'','confound':'no common support'},
]
with (D/'identifiability_matrix.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(ident[0].keys()));w.writeheader();w.writerows(ident)
print('ok')
