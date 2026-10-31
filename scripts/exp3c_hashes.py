import csv,glob,json,hashlib,numpy as np
from pathlib import Path
R=Path('/scratch/lijh/simulate_vla/results/experiment_3_controlled');rows=[];hashes=[]
for f in sorted(glob.glob(str(R/'rollout_v2/*/initial_state.json'))):
 p=Path(f);tag=p.parent.name;parts=tag.split('_');g=parts[0];mode=parts[1];seed=int(parts[2].lstrip('r'))
 d=json.load(open(f));z=np.load(p.parent/'initial_observation.npz')
 rows.append({'group':g,'mode':mode,'seed':seed,'q':';'.join(f'{x:.7f}' for x in d['q']),'dq':';'.join(f'{x:.6f}' for x in d['dq']),
  'ee_x':d['ee'][0],'ee_y':d['ee'][1],'ee_z':d['ee'][2],'ee_quat':';'.join(f'{x:.6f}' for x in d['ee_quat']),
  'obj_x':d['obj_pos'][0],'obj_y':d['obj_pos'][1],'obj_z':d['obj_pos'][2],'obj_quat':';'.join(f'{x:.6f}' for x in d['obj_quat']),
  'grip_target':d['grip_target'],'q_target':';'.join(f'{x:.7f}' for x in d['q_target'])})
 hashes.append({'group':g,'mode':mode,'seed':seed,'top_sha':hashlib.sha256(z['top'].tobytes()).hexdigest()[:16],'wrist_sha':hashlib.sha256(z['wrist'].tobytes()).hexdigest()[:16]})
with open(R/'data/initial_states_v2.csv','w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
with open(R/'data/observation_hashes_v2.csv','w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(hashes[0].keys()));w.writeheader();w.writerows(hashes)
print('ok')
