import csv,glob,json
from pathlib import Path
ROOT=Path('/scratch/lijh/simulate_vla');BASE=ROOT/'results/experiment_2_9/c3_home_width_10';OUT=BASE/'data';OUT.mkdir(parents=True,exist_ok=True)
rows=[]
for f in sorted(glob.glob(str(BASE/'runs/*/frozen/c3_home_g*/summary.json'))):
 d=Path(f).parent;s=json.loads(Path(f).read_text());tag=d.name;w=int(tag.split('_g')[1]);rep=Path(f).parts[-4].split('_r')[-1]
 ph=list(csv.DictReader((d/'physics_trace.csv').open(encoding='utf-8-sig')));last=ph[-1]
 first=None
 for c in csv.DictReader((d/'contact_trace.csv').open(encoding='utf-8-sig')):
  if c.get('bodyA')=='1' or c.get('bodyB')=='1':first=c;break
 meta=json.loads((d/'metadata.json').read_text());g=meta.get('gripper_init',{})
 rows.append({'run':tag,'rep':rep,'width_mm':w,'gripper_actual_m':g.get('actual_m'),'obj_move_mm':g.get('obj_move_mm'),
              'success':bool(s['success']),'max_lift_mm':float(s['max_lift_m'])*1000,'wall_contact':bool(s['wall_contact']),
              'final_gap_mm':float(last['obj_wall_gap_m'])*1000,'final_dy_mm':float(last['obj_dy_m'])*1000,
              'final_tilt_deg':float(last['obj_tilt_deg']),
              'first_link':(first['nameA'] if first and first['bodyA']=='1' else first['nameB']) if first else '',
              'first_time_s':float(first['physics_time_sec']) if first else ''})
with (OUT/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
 wri=csv.DictWriter(f,fieldnames=list(rows[0].keys()));wri.writeheader();wri.writerows(rows)
summary=[]
for w in (14,40):
 sub=[r for r in rows if r['width_mm']==w]
 summary.append({'width_mm':w,'n':len(sub),'success':sum(r['success'] for r in sub),'accuracy':sum(r['success'] for r in sub)/len(sub),
                 'max_lift_mean_mm':sum(r['max_lift_mm'] for r in sub)/len(sub),
                 'max_lift_max_mm':max(r['max_lift_mm'] for r in sub),
                 'final_gap_mean_mm':sum(r['final_gap_mm'] for r in sub)/len(sub),
                 'final_tilt_max_deg':max(r['final_tilt_deg'] for r in sub),
                 'wall_contact':sum(r['wall_contact'] for r in sub)})
with (OUT/'accuracy_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
 wri=csv.DictWriter(f,fieldnames=list(summary[0].keys()));wri.writeheader();wri.writerows(summary)
(BASE/'summary.json').write_text(json.dumps({'snapshot':'C3 object pose + HOME robot','per_width':summary},indent=2))
print(json.dumps(summary,indent=1))
