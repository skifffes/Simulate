import json
import sys

d = json.load(open(sys.argv[1]))
for k, v in d['b90'].items():
    print(f"B90 {k}: last_push_tick={v['last_push_tick']} "
          f"obj_total_dx={round(v['obj_total_dx'], 3)} "
          f"ticks_after={v['ticks_after_stop']} contact_after={v['contact_substeps_after_stop']}")
    print('   rel_when_pushing', [round(x, 3) for x in (v['ref_rel_when_pushing'] or [])])
    print('   rel_after_stop  ', [round(x, 3) for x in (v['mean_rel_after_stop'] or [])])
for k, v in d['a90'].items():
    print(f"A90 {k}: min_dist3d={round(v['min_dist3d'],3)} @tick {v['min_dist_tick']} "
          f"dxyz={[round(x,3) for x in v['min_dx_dy_dz']]} final={round(v['final_dist3d'],3)} "
          f"grip=[{round(v['grip_min'],3)},{round(v['grip_max'],3)}] contact={v['ever_contact']}")
