"""实验 3 v3：continuous transition 的目标到达与稳定验证；未达目标不启动 D2。"""
import argparse,json,sys
from pathlib import Path
import numpy as np,pybullet as p,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from run_b90_controlled import ControlledEpisode
from run_b90_snapshot import SnapshotEpisode
from manipulation import finger_widths,set_gripper

class ConvergedEpisode(ControlledEpisode):
    def __init__(self,args,snap,condition,group,mode):
        self.raw_actions=[]
        super().__init__(args,snap,condition,group,mode)
    def policy_tick(self,skill,task,max_lift_check=None):
        a=super().policy_tick(skill,task,max_lift_check)
        self.raw_actions.append(np.asarray(a,float).copy())
        return a
    def build(self):
        super().build()
        qb=np.array([float(x) for x in next(x for x in __import__('csv').DictReader(open(ROOT/'results/experiment_3/data/b_training_start_states.csv',encoding='utf-8-sig')) if x['b_candidate_id']=='ep064_f000')['q'].split(';')])
        gw=0.035;self.qb=qb;self.gw=gw
        # continue holding qb until convergence; not fixed 30 steps
        good=0;max_steps=1200;log=[]
        for i in range(max_steps):
            self.sim.command(qb);set_gripper(self.sim,gw,20.0,0.05)
            self.sim.step();self.physics_step+=1
            q=self.sim.q();dq=np.array([p.getJointState(self.sim.robot,j,physicsClientId=self.sim.client)[1] for j in range(9)])
            pos,_=p.getBasePositionAndOrientation(self.block,physicsClientId=self.sim.client);vel,_=p.getBaseVelocity(self.block,physicsClientId=self.sim.client)
            err=float(np.linalg.norm(q-qb));mx=float(np.max(np.abs(q-qb)));dqn=float(np.linalg.norm(dq));objv=float(np.linalg.norm(vel));grip=abs(float(np.mean(finger_widths(self.sim)))-gw)
            ok=(err<=0.01 and mx<=0.01 and dqn<=0.2 and objv<=0.01 and grip<=0.002)
            good=good+1 if ok else 0
            if i%20==0 or ok:
                log.append({'step':i,'q_err':err,'q_max':mx,'dq_norm':dqn,'obj_speed':objv,'grip_err_mm':grip*1000,'ok':ok,'good_count':good})
            if good>=120:
                self.converged=True;self.conv_steps=i+1;break
        else:
            self.converged=False;self.conv_steps=max_steps
        self.converged_state={'steps':self.conv_steps,'q_err':err,'q_max':mx,'dq_norm':dqn,'obj_speed':objv,'grip_err_mm':grip*1000,'converged':self.converged}
        import csv
        (self.out/'transition_convergence.csv').parent.mkdir(parents=True,exist_ok=True)
        with (self.out/'transition_convergence.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(log[0].keys()));w.writeheader();w.writerows(log)
        # final state at first inference (or failure)
        q=self.sim.q();dq=np.array([p.getJointState(self.sim.robot,j,physicsClientId=self.sim.client)[1] for j in range(9)])
        pos,quat=p.getBasePositionAndOrientation(self.block,physicsClientId=self.sim.client);vel,ang=p.getBaseVelocity(self.block,physicsClientId=self.sim.client)
        ee=p.getLinkState(self.sim.robot,11,computeForwardKinematics=True,physicsClientId=self.sim.client)
        self.initial_state={'q':q.tolist(),'dq':dq.tolist(),'ee':list(ee[4]),'ee_quat':list(ee[5]),'obj_pos':list(pos),'obj_quat':list(quat),'obj_vel':list(vel),'obj_ang':list(ang),'seed':self.control_seed,'group':self.group,'mode':self.mode,'q_target':qb.tolist(),'grip_target':gw,'converged':self.converged,'conv_steps':self.conv_steps}
        (self.out/'initial_state.json').write_text(json.dumps(self.initial_state,indent=2))
        obs=self.rig256.capture();np.savez_compressed(self.out/'initial_observation.npz',top=obs['top'],wrist=obs['wrist'])
    def run(self):
        self.build()
        if not self.converged:
            self.finish(True,False,None)
            if p.isConnected(self.sim.client):p.disconnect(self.sim.client)
            return
        seed=self.control_seed;torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed)
        success,first=(False,None)
        try:success,first=self.run_b()
        finally:
            np.savez_compressed(self.out/'raw_actions.npz',actions=np.asarray(self.raw_actions))
            self.finish(True,success,first)
            if p.isConnected(self.sim.client):p.disconnect(self.sim.client)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--group',choices=['A','B','C'],required=True);ap.add_argument('--mode',choices=['continuous'],default='continuous');ap.add_argument('--snap',type=Path,required=True);ap.add_argument('--policy',required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--gripper-width',type=float,default=None);ap.add_argument('--max-ticks-b',type=int,default=260);ap.add_argument('--checkpoint-sha',default='34a6e308');ap.add_argument('--commit',default='b7d0a28')
    a=ap.parse_args();snap=json.loads(a.snap.read_text());ep=ConvergedEpisode(a,snap,'frozen',a.group,'continuous');ep.run()
if __name__=='__main__':main()
