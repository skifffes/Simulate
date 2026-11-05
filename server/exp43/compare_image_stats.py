import json
import glob

import cv2
import numpy as np

# 新数据：抽 3 个不同条件各 1 帧
NEW = [
    '/scratch/lijh/exp43/data_a90a/A90/videos/chunk-000/observation.images.top/episode_000000.mp4',
    '/scratch/lijh/exp43/data_b90a/B90/videos/chunk-000/observation.images.top/episode_000000.mp4',
    '/scratch/lijh/exp43/data_b90b/B90/videos/chunk-000/observation.images.top/episode_000000.mp4',
]
for v in NEW:
    cap = cv2.VideoCapture(v)
    ok, f = cap.read()
    cap.release()
    print('new', v.split('/')[3], v.split('/')[-3] if 'images' in v else '', ok,
          round(float(f.mean()), 1) if ok else None, round(float(f.std()), 1) if ok else None)

# 旧数据：读 npz 里的 top 图像（npy 目录）
meta = [json.loads(l) for l in open('/scratch/lijh/exp42/data/meta.jsonl', encoding='utf-8')
        if l.strip()]
m = meta[0]
arr = np.load('/scratch/lijh/exp42/data/' + m['top_path'].replace('data/', ''))
print('old', m['skill'], m['top_path'], arr.shape, 'mean', round(float(arr.mean()), 1),
      'std', round(float(arr.std()), 1))
