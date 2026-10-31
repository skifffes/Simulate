"""实验 3 受控诊断：带显式配对种子的冻结 D2 B90 runner。"""
import argparse,json,sys
from pathlib import Path
import numpy as np,pybullet as p,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from run_b90_snapshot import SnapshotEpisode

class SeededEpisode(SnapshotEpisode):
    def __init__(self,args,snap,condition):
        self.control_seed=int(args.seed)
        super().__init__(args,snap,condition)
    def run(self):
        self.build()
        seed=int(self.control_seed)
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        q=self.sim.q();dq=np.array([p.getJointState(self.sim.robot,j,physicsClientId=self.sim.client)[1] for j in range(9)])
        pos,quat=p.getBasePositionAndOrientation(self.block,physicsClientId=self.sim.client)
        vel,ang=p.getBaseVelocity(self.block,physicsClientId=self.sim.client)
        ee=p.getLinkState(self.sim.robot,11,computeForwardKinematics=True,physicsClientId=self.sim.client)
        init={'q':q.tolist(),'dq':dq.tolist(),'ee':list(ee[4]),'ee_quat':list(ee[5]),'obj_pos':list(pos),'obj_quat':list(quat),'obj_vel':list(vel),'obj_ang':list(ang),'seed':seed,'snapshot':self.snap['bullet'],'tag':self.snap.get('tag')}
        (self.out/'initial_state.json').write_text(json.dumps(init,indent=2))
        success,first=(False,None)
        try:
            success,first=self.run_b()
        finally:
            self.finish(True,success,first)
            if p.isConnected(self.sim.client):p.disconnect(self.sim.client)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--snap',type=Path,required=True);ap.add_argument('--policy',required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--gripper-width',type=float,default=None);ap.add_argument('--max-ticks-b',type=int,default=260);ap.add_argument('--checkpoint-sha',default='34a6e308');ap.add_argument('--commit',default='576048a')
    a=ap.parse_args();snap=json.loads(a.snap.read_text());ep=SeededEpisode(a,snap,'frozen');ep.run()
if __name__=='__main__':main()
