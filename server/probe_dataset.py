"""在服务器 venv 中快速自检数据集与 smolvla 策略是否可用。"""

import os

from lerobot.datasets.lerobot_dataset import LeRobotDataset

root = os.path.expanduser("~/simulate_vla/data/lerobot_panda_pick_place")
ds = LeRobotDataset("local/panda_pick_place", root=root, video_backend="pyav")
print("episodes", ds.num_episodes, "frames", ds.num_frames, "fps", ds.fps)
print("stats keys", sorted(ds.meta.stats.keys()))
item = ds[0]
for key in ("observation.state", "action", "observation.images.top", "observation.images.wrist", "task"):
    value = item.get(key)
    print(" ", key, getattr(value, "shape", value))
print("state mean", ds.meta.stats["observation.state"]["mean"])
print("action mean", ds.meta.stats["action"]["mean"])
