"""实验 3 v4：首推理状态与真实模型输入的哈希归档。"""
import csv,glob,json,hashlib,numpy as np
from pathlib import Path
ROOT=Path('/scratch/lijh/simulate_vla');R=ROOT/'results/experiment_3_controlled';rows=[]
for f in sorted(glob.glob(str(R/'state_audit_v4/*/initial_state.json'))):
    p=Path(f);tag=p.parent.name;g=tag[0];seed=int(tag.split('_')[-1].lstrip('r'));d=json.load(open(f));z=np.load(p.parent/'model_input.npz')
    rows.append({'group':g,'seed':seed,'converged':d['converged'],'conv_steps':d['conv_steps'],
                 'q_err_to_qb':float(np.linalg.norm(np.array(d['q'])-np.array(d['q_target']))),'dq_norm':float(np.linalg.norm(d['dq'])),
                 'finger_left_mm':d['grip_actual'][0]*1000,'finger_right_mm':d['grip_actual'][1]*1000,
                 'finger_dq_l2':float(np.linalg.norm(d['finger_dq'])),
                 'contacts_start':json.dumps(d.get('contacts_start',{})),
                 'top_sha':hashlib.sha256(z['top'].tobytes()).hexdigest()[:16],
                 'wrist_sha':hashlib.sha256(z['wrist'].tobytes()).hexdigest()[:16],
                 'state_sha':hashlib.sha256(z['state'].tobytes()).hexdigest()[:16]})
with open(R/'data/observation_hashes_v4.csv','w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
print('ok')
