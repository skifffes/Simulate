"""固定时间步的 Panda 场景；所有正式动作使用电机控制。"""
from pathlib import Path
import time
import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image

HOME = [0, -0.5, 0, -2.1, 0, 1.65, 0.78]


class Simulation:
    def __init__(self, gui=False, realtime=False, speed=1.0, reference_cube=True):
        self.gui, self.realtime = gui, realtime
        self.speed = max(1e-3, float(speed))
        # 0B 用固定立方体作视觉参考；0C 需要自己的可移动工件，因此可关闭。
        self.reference_cube = reference_cube
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        if self.client < 0:
            raise RuntimeError('PyBullet connection failed')
        self.dt = 1 / 240
        self.reset()

    def reset(self):
        p.resetSimulation(physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(self.dt, physicsClientId=self.client)
        p.setPhysicsEngineParameter(deterministicOverlappingPairs=1, numSolverIterations=100, physicsClientId=self.client)
        self.plane = p.loadURDF('plane.urdf', physicsClientId=self.client)
        self.cube = (p.loadURDF('cube_small.urdf', [0.55, 0, 0.025], useFixedBase=True,
                                physicsClientId=self.client) if self.reference_cube else -1)
        # 抓取任务中手指与工件的接触是预期接触，不属于碰撞失败。
        self.allowed_contacts = set()
        self.robot = p.loadURDF('franka_panda/panda.urdf', useFixedBase=True,
                                flags=p.URDF_USE_SELF_COLLISION, physicsClientId=self.client)
        # 该 URDF 的 panda_link7 与 panda_hand 碰撞几何常年重叠约 2.5 cm。
        # 失败判定本来就把这一固定连接对排除在外，但物理引擎仍按罚函数求解，
        # 产生约 19 kN 接触力，相当于给 panda_joint5 附加约 12 N·m 制动，
        # 曾使 HOME→C 出现假 tracking_error。这里在动力学层面同样关闭该对。
        self.ignored_contacts = [(6, 8)]
        for link_a, link_b in self.ignored_contacts:
            p.setCollisionFilterPair(self.robot, self.robot, link_a, link_b, enableCollision=0,
                                     physicsClientId=self.client)
        self.joints = list(range(7))
        self.ee = 11  # panda_grasptarget：夹爪中心参考坐标系。
        self.lower = np.array([p.getJointInfo(self.robot, j, physicsClientId=self.client)[8] for j in self.joints])
        self.upper = np.array([p.getJointInfo(self.robot, j, physicsClientId=self.client)[9] for j in self.joints])
        # 只在每次试验初始化时设置初始关节，技能执行中禁止瞬移。
        for j, q in enumerate(HOME):
            p.resetJointState(self.robot, j, q, physicsClientId=self.client)
        for j in [9, 10]:
            p.resetJointState(self.robot, j, 0.035, physicsClientId=self.client)
            p.setJointMotorControl2(self.robot, j, p.POSITION_CONTROL, 0.035, force=30, physicsClientId=self.client)
        self.command(HOME)
        p.resetDebugVisualizerCamera(1.3, 45, -28, [0.35, 0, 0.3], physicsClientId=self.client)
        self.steps = 0
        for _ in range(120):
            self.step()

    def command(self, q):
        p.setJointMotorControlArray(self.robot, self.joints, p.POSITION_CONTROL,
                                   targetPositions=list(q), forces=[87, 87, 87, 87, 12, 12, 12],
                                   positionGains=[0.2] * 7, physicsClientId=self.client)

    def q(self):
        return np.array([s[0] for s in p.getJointStates(self.robot, self.joints, physicsClientId=self.client)])

    def pose(self):
        state = p.getLinkState(self.robot, self.ee, computeForwardKinematics=True, physicsClientId=self.client)
        return np.array(state[4]), np.array(state[5])

    def step(self):
        p.stepSimulation(physicsClientId=self.client)
        self.steps += 1
        if self.gui:
            text = 'q(rad): ' + ' '.join(f'{v:.2f}' for v in self.q())
            self.label = p.addUserDebugText(text, [0, 0, 0.95], textSize=1.1,
                                          replaceItemUniqueId=getattr(self, 'label', -1), physicsClientId=self.client)
        if self.realtime:
            time.sleep(self.dt / self.speed)

    def collisions(self, allowed=None):
        """返回非预期碰撞。

        allowed 为 {(body_id, robot_link_index)} 集合，表示允许的接触，
        例如 0C 中手指与工件的接触。默认使用 self.allowed_contacts。
        """
        allowed = self.allowed_contacts if allowed is None else allowed
        contacts = p.getContactPoints(bodyA=self.robot, physicsClientId=self.client)
        bad = []
        for c in contacts:
            if c[8] >= -0.0001:
                continue
            # 固定底座与地面的接触属于预期支撑，不计为失败。
            if c[2] == self.plane and c[3] == -1:
                continue
            if (c[2], c[3]) in allowed:
                continue
            # URDF 固定连接的相邻夹爪链接可能具有重叠碰撞几何。
            if c[2] == self.robot:
                a, b = c[3], c[4]
                pa = p.getJointInfo(self.robot, a, physicsClientId=self.client)[16] if a >= 0 else -2
                pb = p.getJointInfo(self.robot, b, physicsClientId=self.client)[16] if b >= 0 else -2
                if a == pb or b == pa or {a, b} == {6, 8}:
                    continue
            bad.append({'body': c[2], 'link_a': c[3], 'link_b': c[4], 'penetration_m': c[8]})
        return bad

    def screenshot(self, path, target=(0.35, 0, 0.3), distance=1.3, yaw=45, pitch=-28,
                   width=960, height=720):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        view = p.computeViewMatrixFromYawPitchRoll(list(target), distance, yaw, pitch, 0, 2)
        proj = p.computeProjectionMatrixFOV(55, width / height, 0.02, 5)
        pixels = p.getCameraImage(width, height, view, proj, renderer=p.ER_TINY_RENDERER, physicsClientId=self.client)[2]
        Image.fromarray(np.asarray(pixels, dtype=np.uint8).reshape(height, width, 4)).convert('RGB').save(path)

    def close(self):
        if p.isConnected(self.client):
            p.disconnect(self.client)
