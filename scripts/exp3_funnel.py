import csv
from pathlib import Path
B=Path(r'D:\python\simulate\results\experiment_3');D=B/'data'
rows=[
 {'stage':1,'metric':'unique A90 terminal states','count':1,'denominator':5,'note':'5 snapshots identical q/object'},
 {'stage':2,'metric':'B90 training starts','count':60,'denominator':60,'note':'yaw90'},
 {'stage':3,'metric':'scene-compatible pairs','count':60,'denominator':60,'note':'object diff 1.5-17.9mm; axis<10deg'},
 {'stage':4,'metric':'goal collision-valid (pilot)','count':3,'denominator':3,'note':'selected 3 B starts'},
 {'stage':5,'metric':'planned (pilot)','count':3,'denominator':3,'note':'collision-checked straight edge; RRT fallback unused'},
 {'stage':6,'metric':'executed (pilot)','count':3,'denominator':3,'note':'dense interpolated joint commands'},
 {'stage':7,'metric':'robot-state matched','count':3,'denominator':3,'note':'q<0.05rad, EE<10mm'},
 {'stage':8,'metric':'visual matched','count':3,'denominator':3,'note':'top MAE<1; wrist<5'},
 {'stage':9,'metric':'B90 pair with >=1 success','count':2,'denominator':3,'note':'AtoB 3/9 rollouts vs Bref 6/9'},
]
with (D/'composability_funnel.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
print('ok')
