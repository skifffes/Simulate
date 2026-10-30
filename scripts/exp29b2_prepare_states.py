"""实验 2.9-B2：真正的 Low-Near / Low-Far / Mid-Near 起点。"""
from __future__ import annotations
import csv, json, math, sys
from pathlib import Path
import numpy as np
import pybullet as p

ROOT = Path('/scratch/lijh/simulate_vla'); sys.path.insert(0, str(ROOT/'src'))
from vla.atomic_tasks.scene2 import build_scene, PLACE, grasp_quat, block_yaw
from vla.atomic_tasks.starts import home_array, ik_solve, set_robot_state, ee_position
from manipulation import finger_widths, set_gripper

OUT = ROOT/'results/experiment_2_9/b2_low_near'; STATES=OUT/'states'; STATES.mkdir(parents=True,exist_ok=True)
DATA=OUT/'data'; DATA.mkdir(parents=True,exist_ok=True)
WIDTHS=[0.014,0.035,0.040]

def width_check(sim, block, q, w):
    set_robot_state(sim, q, finger=0.035); sim.command(q); set_gripper(sim,0.035,20,0.05)
    for _ in range(10): sim.step()
    b0,_=p.getBasePositionAndOrientation(block,physicsClientId=sim.client)
    for j in (9,10): p.resetJointState(sim.robot,j,w,physicsClientId=sim.client)
    sim.command(q); set_gripper(sim,w,20,0.05)
    for _ in range(10): sim.step()
    b1,_=p.getBasePositionAndOrientation(block,physicsClientId=sim.client)
    move=float(np.linalg.norm(np.asarray(b1)-np.asarray(b0))*1000)
    contacts=p.getContactPoints(bodyA=sim.robot,bodyB=block,physicsClientId=sim.client)
    actual=float(np.mean(finger_widths(sim)))
    return {'width_m':w,'actual_m':actual,'obj_move_mm':move,'block_contacts':len(contacts),
            'valid':bool(move<0.5 and not contacts and abs(actual-w)<0.001)}

def main():
    specs = [('low_near',-0.08,0.075),('low_near',-0.09,0.085),('low_near',-0.07,0.070),
             ('low_far',-0.18,0.075),('low_far',-0.19,0.085),
             ('mid_near',-0.08,0.165),('mid_near',-0.09,0.180)]
    rows=[]; counts={}
    for k,(kind,dx,z) in enumerate(specs):
        seed=93000000+k*13
        rng=np.random.default_rng(seed); yaw=math.pi/2+float(rng.uniform(-0.07,0.07))
        sim,block,wall,tray,info=build_scene(seed=seed,start=PLACE,jitter=0.012,yaw=yaw)
        try:
            bp,_=p.getBasePositionAndOrientation(block,physicsClientId=sim.client);bp=np.asarray(bp,float)
            target=np.array([bp[0]+dx, bp[1]+float(rng.uniform(-0.035,0.035)), z])
            q,ok=ik_solve(sim,target,grasp_quat(0.0),home_array())
            if not ok: print('reject',kind,k,'ik');continue
            set_robot_state(sim,q,0.035); sim.command(q); set_gripper(sim,0.035,20,0.05)
            for _ in range(20): sim.step()
            ee=ee_position(sim); err=float(np.linalg.norm(ee-target))
            b0,_=p.getBasePositionAndOrientation(block,physicsClientId=sim.client)
            c0=p.getContactPoints(bodyA=sim.robot,bodyB=block,physicsClientId=sim.client)
            if err>0.03 or c0:
                print('reject',kind,k,'err',round(err,4),'contacts',len(c0));continue
            checks=[width_check(sim,block,q,w) for w in WIDTHS]
            if not all(x['valid'] for x in checks):
                print('reject',kind,k,'width',[(x['width_m'],x['valid'],round(x['obj_move_mm'],2)) for x in checks]);continue
            set_robot_state(sim,q,0.035); sim.command(q); set_gripper(sim,0.035,20,0.05)
            for _ in range(10): sim.step()
            sid=f"{kind}_{counts.get(kind,0):02d}"; counts[kind]=counts.get(kind,0)+1
            bullet=STATES/f'{sid}.bullet'; p.saveBullet(str(bullet),physicsClientId=sim.client)
            obj,oq=p.getBasePositionAndOrientation(block,physicsClientId=sim.client)
            row={'state_id':sid,'kind':kind,'seed':seed,'q':';'.join(f'{v:.7f}' for v in q),
                 'ee_x':float(ee[0]),'ee_y':float(ee[1]),'ee_z':float(ee[2]),
                 'obj_x':float(obj[0]),'obj_y':float(obj[1]),'obj_z':float(obj[2]),
                 'obj_yaw_rad':float(block_yaw(block,sim.client)),
                 'ee_obj_dx':float(ee[0]-obj[0]),'ee_obj_dy':float(ee[1]-obj[1]),
                 'ee_obj_dz':float(ee[2]-obj[2]),'ik_error_m':err,
                 'bullet':str(bullet),'valid':True}
            rows.append(row);print('ACCEPT',sid,kind,'z',round(ee[2],3),'dx',round(row['ee_obj_dx'],3),flush=True)
        finally: p.disconnect(sim.client)
    with (DATA/'state_manifest.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
    for r in rows:
        d={'seed':int(r['seed']),'bullet':r['bullet'],'group':'yaw90','tag':r['state_id'],
           'state_id':r['state_id'],'kind':r['kind'],'source':'exp29b2_generated',
           'obj':[float(r['obj_x']),float(r['obj_y']),float(r['obj_z'])],'obj_yaw':float(r['obj_yaw_rad'])}
        (STATES/f"{r['state_id']}.json").write_text(json.dumps(d,indent=2))
    print('total',len(rows))

if __name__=='__main__':main()
