import csv, json, math
from pathlib import Path
import numpy as np
ROOT=Path('/scratch/lijh/simulate_vla'); BASE=ROOT/'results/experiment_2_9/ab/runs'
OUT=ROOT/'results/experiment_2_9/b2_low_near/data'; OUT.mkdir(parents=True,exist_ok=True)
runs={
 'D2_14':BASE/'D2/low_00_g014_r0/frozen/low_00_g014',
 'D2_35_r0':BASE/'D2/low_00_g035_r0/frozen/low_00_g035',
 'D2_35_r1':BASE/'D2/low_00_g035_r1/frozen/low_00_g035',
 'E1_14':BASE/'E1/low_00_g014_r0/expert_e1/low_00_g014',
 'E1_35':BASE/'E1/low_00_g035_r0/expert_e1/low_00_g035',
}
summ=[]; traj=[]
for name,d in runs.items():
    ph=list(csv.DictReader((d/'physics_trace.csv').open(encoding='utf-8-sig')))
    first=None
    for c in csv.DictReader((d/'contact_trace.csv').open(encoding='utf-8-sig')):
        if c.get('bodyA')=='1' or c.get('bodyB')=='1': first=c; break
    ct=float(first['physics_time_sec']) if first else None
    cz=float(next(r for r in ph if abs(float(r['physics_time_sec'])-ct)<1e-9)['ee_z']) if ct else None
    z=np.array([float(r['ee_z']) for r in ph]); x=np.array([float(r['ee_x']) for r in ph]); t=np.array([float(r['physics_time_sec']) for r in ph])
    pre=ph if ct is None else [r for r in ph if float(r['physics_time_sec'])<=ct]
    zpre=min(float(r['ee_z']) for r in pre)
    # path length until contact
    xs=np.array([float(r['ee_x']) for r in pre]); zs=np.array([float(r['ee_z']) for r in pre])
    path=float(np.sum(np.sqrt(np.diff(xs)**2+np.diff(zs)**2))) if len(xs)>1 else 0
    z05=float(next(r for r in ph if abs(float(r['physics_time_sec'])-0.5)<0.01)['ee_z'])
    z10=float(next(r for r in ph if abs(float(r['physics_time_sec'])-1.0)<0.01)['ee_z'])
    summ.append({'run':name,'contact_time_s':ct,'contact_ee_z_mm':cz*1000 if cz else '',
                 'z_0.5s_mm':z05*1000,'z_1.0s_mm':z10*1000,'min_precontact_z_mm':zpre*1000,'path_len_m':path,
                 'final_x_mm':x[-1]*1000,'final_z_mm':z[-1]*1000})
    for k in range(0,len(ph),12):
        r=ph[k];traj.append({'run':name,'t':r['physics_time_sec'],'ee_x':r['ee_x'],'ee_z':r['ee_z'],'grip':r['grip_actual'],'tilt':r['obj_tilt_deg'],'gap':r['obj_wall_gap_m']})
with (OUT/'original_low00_recheck.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(summ[0].keys()));w.writeheader();w.writerows(summ)
with (OUT/'original_low00_trajectories.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(traj[0].keys()));w.writeheader();w.writerows(traj)
print(json.dumps(summ,indent=1))
