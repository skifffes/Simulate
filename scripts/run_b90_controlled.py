"""实验 3 受控诊断：统一 30-step 稳定、首推理观测保存、连续 transition 版本。"""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np,pybullet as p,torch
from lerobot.utils.control_utils import predict_action
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from run_b90_snapshot import SnapshotEpisode
from run_experiment_2_4 import SPA
from manipulation import set_gripper,finger_widths
from vla.atomic_tasks.scene2 import build_scene
from vla.atomic_tasks.config2 import CAMERAS,IMAGE_SIZE
from vla.cameras import CameraRig
from run_experiment_2_4 import make_view,PolicyRunner,VIDEO_W,VIDEO_H

class ControlledEpisode(SnapshotEpisode):
    def __init__(self,args,snap,condition,group,mode):
        self.control_seed=int(args.seed);self.group=group;self.mode=mode
        super().__init__(args,snap,condition)
        self.out=Path(args.out)/f'{group}_{mode}_r{self.control_seed}'
        self.out.mkdir(parents=True,exist_ok=True)
    def build(self):
        # build in the A terminal scene for continuous B/C
        if self.mode=='continuous':
            A=next(csv.DictReader(open(ROOT/'results/experiment_3/data/a_terminal_states.csv',encoding='utf-8-sig')))
            self.sim,self.block,self.wall,self.tray,self.info=build_scene(seed=int(A['source_seed']),start=(0.55,0.18,0.02),jitter=0,yaw=0.0)
            self.names={self.sim.plane:'table',self.sim.robot:'robot',self.block:'block',self.wall:'wall'}
            for t in self.tray:self.names[t]='tray'
            self.rig256=CameraRig(self.sim,CAMERAS,IMAGE_SIZE,supersample=1)
            self.rig640=CameraRig(self.sim,CAMERAS,(VIDEO_W,VIDEO_H),supersample=1)
            self.side_view,self.side_proj=make_view((0.55,-0.62,0.30),(0.60,0.16,0.10),fov=46.0)
            p.restoreState(fileName=str(ROOT/A['bullet']),physicsClientId=self.sim.client)
            qA=self.sim.q();self.sim.command(qA)
            if self.group=='C':
                path=np.load(ROOT/'results/experiment_3/trajectories/planned/a90_40000000__ep064_f000.npy')
                dense=[]
                for i in range(len(path)-1):
                    a=np.asarray(path[i]);b=np.asarray(path[i+1]);n=max(2,int(np.ceil(np.linalg.norm(b-a)/0.01)))
                    for t in np.linspace(0,1,n,endpoint=False):dense.append(a*(1-t)+b*t)
                dense.append(np.asarray(path[-1]))
                for q in dense:
                    self.sim.command(q)
                    for _ in range(SPA):self.sim.step();self.physics_step+=1
            qb=np.array([float(x) for x in next(csv.DictReader(open(ROOT/'results/experiment_3/data/b_training_start_states.csv',encoding='utf-8-sig')))['q'].split(';')])
            # explicit qB from ep064
            qb=np.array([float(x) for x in next(x for x in csv.DictReader(open(ROOT/'results/experiment_3/data/b_training_start_states.csv',encoding='utf-8-sig')) if x['b_candidate_id']=='ep064_f000')['q'].split(';')])
            gw=0.035
            if self.group=='B':
                for j in range(7):p.resetJointState(self.sim.robot,j,float(qb[j]),physicsClientId=self.sim.client)
            self.sim.command(qb);set_gripper(self.sim,gw,20.0,0.05)
            for _ in range(30):self.sim.step();self.physics_step+=1
        else:
            self.sim,self.block,self.wall,self.tray,self.info=build_scene(seed=self.seed,start=(0.55,0.18,0.02),jitter=0,yaw=0.0)
            self.names={self.sim.plane:'table',self.sim.robot:'robot',self.block:'block',self.wall:'wall'}
            for t in self.tray:self.names[t]='tray'
            self.rig256=CameraRig(self.sim,CAMERAS,IMAGE_SIZE,supersample=1)
            self.rig640=CameraRig(self.sim,CAMERAS,(VIDEO_W,VIDEO_H),supersample=1)
            self.side_view,self.side_proj=make_view((0.55,-0.62,0.30),(0.60,0.16,0.10),fov=46.0)
            p.restoreState(fileName=str(ROOT/self.snap['bullet']),physicsClientId=self.sim.client)
            q=self.sim.q();self.sim.command(q);set_gripper(self.sim,float(finger_widths(self.sim)[0]),20.0,0.05)
            for _ in range(30):self.sim.step();self.physics_step+=1
        st=None
        self.initial_z=float(p.getBasePositionAndOrientation(self.block,physicsClientId=self.sim.client)[0][2])
        self.initial_xy=np.asarray(p.getBasePositionAndOrientation(self.block,physicsClientId=self.sim.client)[0][:2])
        self.b_start_z=self.initial_z
        self.runner=PolicyRunner(self.args.policy)
    def run(self):
        self.build()
        seed=self.control_seed
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed)
        q=self.sim.q();dq=np.array([p.getJointState(self.sim.robot,j,physicsClientId=self.sim.client)[1] for j in range(9)])
        pos,quat=p.getBasePositionAndOrientation(self.block,physicsClientId=self.sim.client)
        vel,ang=p.getBaseVelocity(self.block,physicsClientId=self.sim.client)
        ee=p.getLinkState(self.sim.robot,11,computeForwardKinematics=True,physicsClientId=self.sim.client)
        self.initial_top=self.rig256.capture()
        init={'q':q.tolist(),'dq':dq.tolist(),'ee':list(ee[4]),'ee_quat':list(ee[5]),'obj_pos':list(pos),'obj_quat':list(quat),'obj_vel':list(vel),'obj_ang':list(ang),'seed':seed,'group':self.group,'mode':self.mode,'snapshot':self.snap['bullet'],'q_target':q.tolist(),'grip_target':float(finger_widths(self.sim)[0])}
        (self.out/'initial_state.json').write_text(json.dumps(init,indent=2))
        np.savez_compressed(self.out/'initial_observation.npz',top=self.initial_top['top'],wrist=self.initial_top['wrist'])
        success,first=(False,None)
        try:success,first=self.run_b()
        finally:
            self.finish(True,success,first)
            if p.isConnected(self.sim.client):p.disconnect(self.sim.client)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--group',choices=['A','B','C'],required=True);ap.add_argument('--mode',choices=['snapshot','continuous'],required=True);ap.add_argument('--snap',type=Path,required=True);ap.add_argument('--policy',required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--gripper-width',type=float,default=None);ap.add_argument('--max-ticks-b',type=int,default=260);ap.add_argument('--checkpoint-sha',default='34a6e308');ap.add_argument('--commit',default='576048a')
    a=ap.parse_args();snap=json.loads(a.snap.read_text());ep=ControlledEpisode(a,snap,'frozen',a.group,a.mode);ep.run()
if __name__=='__main__':main()
