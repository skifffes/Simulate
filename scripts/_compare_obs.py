"""对比数据集首帧观测与评测脚本首帧观测，排查 state/动作语义是否一致。"""

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/lerobot_panda_pick_place")
info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
path = root / info["data_path"].format(episode_chunk=0, episode_index=0)
data = pq.read_table(path).to_pydict()
rows = [i for i, ep in enumerate(data["episode_index"]) if int(ep) == 0]
print("episode0 长度", len(rows))
print("state[0]", np.round(np.asarray(data["observation.state"][rows[0]]), 4))
print("state[-1]", np.round(np.asarray(data["observation.state"][rows[-1]]), 4))
print("action[0]", np.round(np.asarray(data["action"][rows[0]]), 4))
print("action[-1]", np.round(np.asarray(data["action"][rows[-1]]), 4))

npz_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results/eval_smoke_3k/episodes/episode_20261021.npz")
if npz_path.exists():
    blob = np.load(npz_path)
    states = blob["states"]
    actions = blob["actions"]
    print("eval state[0]", np.round(states[0], 4))
    print("eval state[-1]", np.round(states[-1], 4))
    print("eval action[0]", np.round(actions[0], 4))
    print("eval action[-1]", np.round(actions[-1], 4))
