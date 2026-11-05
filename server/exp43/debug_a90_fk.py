import json
from pathlib import Path

import numpy as np
import pandas as pd
import pybullet as p

from exp41 import scene41

SRV = Path('/scratch/lijh/exp43')
MAN = Path('/scratch/lijh/exp40/repo/results/experiment_4_3/data_expansion/train_manifest_v2.json')
man = json.loads(MAN.read_text(encoding='utf-8'))
e = next(x for x in man['episodes'] if x['skill'] == 'A90')
df = pd.read_parquet(SRV / e['parquet'])
st = np.stack(df['observation.state'].to_numpy()).astype(float)
ac = np.stack(df['action'].to_numpy()).astype(float)

scene = scene41.Scene41('B90', obj_pos=(0.55, 0.21, 0.0201), obj_yaw=np.pi / 2, robot_q=None)
sim = scene.sim


def fk(q):
    for j in range(7):
        p.resetJointState(sim.robot, j, float(q[j]), targetVelocity=0.0,
                          physicsClientId=sim.client)
    link = p.getLinkState(sim.robot, 11, computeForwardKinematics=True,
                          physicsClientId=sim.client)
    return np.asarray(link[4], dtype=float)


zs = []
for i in range(0, len(df), 10):
    zs.append(round(float(fk(st[i, :7])[2]), 4))
print('EE z every 10 frames:', zs)
print('max-min z:', round(max(zs) - min(zs), 4))
print('close0=56 z:', round(float(fk(st[55, :7])[2]), 4), '-> later max:',
      round(max(float(fk(st[i, :7])[2]) for i in range(56, len(df), 5)), 4))
scene.close()
