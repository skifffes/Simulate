"""实验 3.2：跨 B 起点的 A/B/C 受控 episode。"""
import argparse,csv,json,random,sys
from pathlib import Path
import numpy as np,pybullet as p,torch
ROOT=Path('/scratch/lijh/simulate_vla');sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from run_b90_snapshot import SnapshotEpisode
from run_experiment_2_4 import SPA,PolicyRunner,VIDEO_W,VIDEO_H,make_view
from vla.atomic_tasks.scene2 import build_scene
from vla.cameras import CameraRig
from vla.atomic_tasks.config2 import CAMERAS,IMAGE_SIZE
from manipulation import set_gripper,finger_widths

A=next(csv.DictReader(open(ROOT/'results/experiment_3/data/a_terminal_states.csv',encoding='utf-8-sig')))
BROWS={x['b_candidate_id']:x for x in csv.DictReader(open(ROOT/'results/experiment_3/data/b_training_start_states.csv',encoding='utf-8-sig'))}
qa=np.array([float(x) for x in A['q'].split(';')])

class Exp32Episode(SnapshotEpisode):
    def __init__(self,args,target,group):
        self.target=target;self.exp_group=group;self.group=group;self.control_seed=int(args.seed);self.brow=BROWS[target]
        snap={'seed':int(self.brow['scene_seed']),'bullet':'','group':'yaw90','tag':f'{target}_{group}_r{args.seed}','episode':int(self.brow['episode_id']),'frame':0}
        super().__init__(args,snap,'frozen')
        self.out=Path(args.out)/f'{group}_{target}_r{args.seed}';self.out.mkdir(parents=True,exist_ok=True)
        self.raw_actions=[]
    def _setup(self,sim):
        self.sim=sim;self.block=self.block if hasattr(self,'block') else None
    def build(self):
        br=self.brow;qb=np.array([float(x) for x in br['q'].split(';')]);gw=float(br['grip_mm'])/1000
        if self.exp_group=='A':
            self.sim,self.block,self.wall,self.tray,self.info=build_scene(seed=int(br['scene_seed']),start=(float(br['obj_x']),float(br['obj_y']),float(br['obj_z'])),jitter=0,yaw=float(br['obj_yaw_rad']))
        else:
            self.sim,self.block,self.wall,self.tray,self.info=build_scene(seed=int(A['source_seed']),start=(0.55,0.18,0.02),jitter=0,yaw=0.0)
            p.restoreState(fileName=str(ROOT/A['bullet']),physicsClientId=self.sim.client)
        self.names={self.sim.plane:'table',self.sim.robot:'robot',self.block:'block',self.wall:'wall'}
        for t in self.tray:self.names[t]='tray'
        self.rig256=CameraRig(self.sim,CAMERAS,IMAGE_SIZE,supersample=1)
        self.rig640=CameraRig(self.sim,CAMERAS,(VIDEO_W,VIDEO_H),supersample=1)
        self.side_view,self.side_proj=make_view((0.55,-0.62,0.30),(0.60,0.16,0.10),fov=46.0)
        if self.exp_group=='C':
            pf=ROOT/f'results/experiment_3_2/trajectories/planned/{self.target}.npy'
            path=np.load(pf) if pf.exists() else np.asarray([qa,qb])
            dense=[]
            for i in range(len(path)-1):
                a=np.asarray(path[i]);b=np.asarray(path[i+1]);n=max(2,int(np.ceil(np.linalg.norm(b-a)/0.01)))
                for t in np.linspace(0,1,n,endpoint=False):dense.append(a*(1-t)+b*t)
            dense.append(np.asarray(path[-1]))
            for q in dense:
                self.sim.command(q)
                for _ in range(SPA):self.sim.step();self.physics_step+=1
        else:
            for j in range(7):p.resetJointState(self.sim.robot,j,float(qb[j]),physicsClientId=self.sim.client)
        self.sim.command(qb);set_gripper(self.sim,gw,20.0,0.05)
        # convergence gate
        good=0;max_steps=1200;log=[]
        for i in range(max_steps):
            self.sim.command(qb);set_gripper(self.sim,gw,20.0,0.05);self.sim.step();self.physics_step+=1
            q=self.sim.q();dq=np.array([p.getJointState(self.sim.robot,j,physicsClientId=self.sim.client)[1] for j in range(9)])
            pos,_=p.getBasePositionAndOrientation(self.block,physicsClientId=self.sim.client);vel,_=p.getBaseVelocity(self.block,physicsClientId=self.sim.client)
            err=float(np.linalg.norm(q-qb));mx=float(np.max(np.abs(q-qb)));dqn=float(np.linalg.norm(dq));objv=float(np.linalg.norm(vel));grip=abs(float(np.mean(finger_widths(self.sim)))-gw)
            ok=(err<=0.01 and mx<=0.01 and dqn<=0.2 and objv<=0.01 and grip<=0.002);good=good+1 if ok else 0
            if i%20==0 or ok:log.append({'step':i,'q_err':err,'dq_norm':dqn,'obj_speed':objv,'grip_err_mm':grip*1000,'ok':ok})
            if good>=120:self.converged=True;break
        else:self.converged=False
        self.qb=qb;self.gw=gw;self.conv_steps=i+1
        self.initial_z=float(p.getBasePositionAndOrientation(self.block,physicsClientId=self.sim.client)[0][2]);self.initial_xy=np.asarray(p.getBasePositionAndOrientation(self.block,physicsClientId=self.sim.client)[0][:2]);self.b_start_z=self.initial_z
        ee=p.getLinkState(self.sim.robot,11,computeForwardKinematics=True,physicsClientId=self.sim.client);pos,quat=p.getBasePositionAndOrientation(self.block,physicsClientId=self.sim.client);vel,ang=p.getBaseVelocity(self.block,physicsClientId=self.sim.client)
        fdq=[float(p.getJointState(self.sim.robot,j,physicsClientId=self.sim.client)[1]) for j in (9,10)]
        contacts=p.getContactPoints(physicsClientId=self.sim.client);cs={'robot_object':0,'object_table':0,'object_wall':0,'robot_other':0}
        for c in contacts:
            ba,bb=c[1],c[2]
            if {ba,bb}=={self.sim.robot,self.block}:cs['robot_object']+=1
            elif {ba,bb}=={self.block,self.sim.plane}:cs['object_table']+=1
            elif {ba,bb}=={self.block,self.wall}:cs['object_wall']+=1
            elif self.sim.robot in (ba,bb):cs['robot_other']+=1
        self.initial_state={'q':self.sim.q().tolist(),'dq':dq.tolist(),'q_target':qb.tolist(),'grip_target':gw,'grip_actual':[float(x) for x in finger_widths(self.sim)],'finger_dq':fdq,'ee':list(ee[4]),'ee_quat':list(ee[5]),'obj_pos':list(pos),'obj_quat':list(quat),'obj_vel':list(vel),'obj_ang':list(ang),'contacts_start':cs,'converged':self.converged,'conv_steps':self.conv_steps,'group':self.exp_group,'target':self.target,'seed':self.control_seed}
        (self.out/'initial_state.json').write_text(json.dumps(self.initial_state,indent=2))
        obs=self.rig256.capture();np.savez_compressed(self.out/'initial_observation.npz',top=obs['top'],wrist=obs['wrist'])
        top=torch.from_numpy(np.ascontiguousarray(obs['top'])).float().div_(255.0).permute(2,0,1).numpy();wrist=torch.from_numpy(np.ascontiguousarray(obs['wrist'])).float().div_(255.0).permute(2,0,1).numpy()
        np.savez_compressed(self.out/'model_input.npz',top=top,wrist=wrist,state=np.concatenate([self.sim.q(),finger_widths(self.sim)]).astype(np.float32),task='push the orange block along the positive X direction to the wall')
        import csv as _c
        with (self.out/'transition_convergence.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=_c.DictWriter(f,fieldnames=list(log[0].keys()));w.writeheader();w.writerows(log)
        self.runner=PolicyRunner(self.args.policy) if not getattr(self.args,'state_only',False) else None
    def policy_tick(self,skill,task,max_lift_check=None):
        a=super().policy_tick(skill,task,max_lift_check);self.raw_actions.append(np.asarray(a,float).copy());return a
    def run(self):
        self.build()
        if getattr(self.args,'state_only',False) or not self.converged:
            self.finish(True,False,None)
            if p.isConnected(self.sim.client):p.disconnect(self.sim.client)
            return
        seed=self.control_seed;random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed)
        success,first=(False,None)
        try:success,first=self.run_b()
        finally:
            np.savez_compressed(self.out/'raw_actions.npz',actions=np.asarray(self.raw_actions));self.finish(True,success,first)
            if p.isConnected(self.sim.client):p.disconnect(self.sim.client)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--target',required=True);ap.add_argument('--group',choices=['A','B','C'],required=True);ap.add_argument('--policy',required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--state-only',action='store_true');ap.add_argument('--gripper-width',type=float,default=None);ap.add_argument('--max-ticks-b',type=int,default=260);ap.add_argument('--checkpoint-sha',default='34a6e308');ap.add_argument('--commit',default='915422a')
    a=ap.parse_args();Exp32Episode(a,a.target,a.group).run()
if __name__=='__main__':main()

