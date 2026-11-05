import shutil
import subprocess
import time

print('time', time.strftime('%H:%M:%S'))
for b in ('ffprobe', 'ffmpeg'):
    print(b, shutil.which(b))
try:
    import cv2
    print('cv2', cv2.__version__)
except Exception as e:  # noqa: BLE001
    print('cv2 missing', e)
try:
    import pyarrow
    print('pyarrow', pyarrow.__version__)
except Exception as e:  # noqa: BLE001
    print('pyarrow missing', e)
out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                      '-show_entries', 'stream=nb_frames,width,height,r_frame_rate',
                      '-of', 'csv=p=0',
                      '/scratch/lijh/exp43/data_a90a/A90/videos/chunk-000/'
                      'observation.images.top/episode_000000.mp4'],
                     capture_output=True, text=True)
print('ffprobe rc', out.returncode, out.stdout.strip(), out.stderr.strip()[:200])
