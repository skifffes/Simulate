import json
from pathlib import Path
out = Path('results/experiment_2_6/B90_gripper_extension/states')
out.mkdir(parents=True, exist_ok=True)
seen = {}
for rep in [
    'results/experiment_2_6/A_midstate_launch/reconstructed/reconstruction_report.json',
    'results/experiment_2_6/B90_gripper_extension/reconstructed/reconstruction_report.json',
]:
    for r in json.loads(Path(rep).read_text()):
        if r['episode'] in seen:
            continue
        seen[r['episode']] = 1
        tag = f"ep{r['episode']:03d}_f{r['frame']:03d}"
        d = {'source': 'D2_B90_midstate', 'tag': tag, 'seed': r['seed'],
             'episode': r['episode'], 'frame': r['frame'], 'group': r['group'],
             'mode': r['mode'], 'bullet': r['bullet'],
             'q_error_rad': r['q_error_rad'], 'gripper_error_m': r['gripper_error_m']}
        (out / (tag + '.json')).write_text(json.dumps(d, indent=2))
        print(tag)
