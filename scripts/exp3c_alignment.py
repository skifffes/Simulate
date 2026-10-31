import csv,math,numpy as np
from pathlib import Path
R=Path(r'D:\python\simulate\results\experiment_3_controlled');D=R/'data'
rows=list(csv.DictReader((D/'initial_states.csv').open(encoding='utf-8-sig')));M={r['state']:r for r in rows}
def qa(q):return np.array([float(x) for x in q.split(';')])
def quat_angle(a,b):
 a=np.array([float(x) for x in a.split(';')]);b=np.array([float(x) for x in b.split(';')]);d=abs(float(np.dot(a,b)));return math.degrees(2*math.acos(min(1,d)))
def yaw_axis(q):
 x,y,z,w=[float(t) for t in q.split(';')];return math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
def axis_angle(a,b):
 d=abs(math.atan2(math.sin(yaw_axis(a)-yaw_axis(b)),math.cos(yaw_axis(a)-yaw_axis(b))));return math.degrees(min(d,math.pi-d))
def arr(r,k):return np.array([float(r[k[0]]),float(r[k[1]]),float(r[k[2]])])
def diff(a,b):
 A=M[a];B=M[b];out={}
 out['q_l2_rad']=float(np.linalg.norm(qa(A['q'])-qa(B['q'])));out['q_max_rad']=float(np.max(np.abs(qa(A['q'])-qa(B['q']))))
 out['dq_l2']=float(np.linalg.norm(qa(A['dq'])-qa(B['dq'])))
 out['ee_pos_mm']=float(np.linalg.norm(arr(A,('ee_x','ee_y','ee_z'))-arr(B,('ee_x','ee_y','ee_z')))*1000)
 out['ee_rot_deg']=quat_angle(A['ee_quat'],B['ee_quat'])
 out['grip_mm']=abs(float(A['grip_left'])+float(A['grip_right'])-float(B['grip_left'])-float(B['grip_right']))/2*1000
 out['obj_pos_mm']=float(np.linalg.norm(arr(A,('obj_x','obj_y','obj_z'))-arr(B,('obj_x','obj_y','obj_z')))*1000)
 out['obj_rot_deg']=axis_angle(A['obj_quat'],B['obj_quat'])
 out['obj_speed_mm_s']=float(np.linalg.norm([float(A['obj_vx'])-float(B['obj_vx']),float(A['obj_vy'])-float(B['obj_vy']),float(A['obj_vz'])-float(B['obj_vz'])])*1000)
 out['ee_obj_rel_mm']=out['obj_pos_mm']
 out['ee_obj_axis_deg']=axis_angle(A['obj_quat'],B['obj_quat'])
 out['contacts_delta']=abs(int(A['n_contacts'])-int(B['n_contacts']))
 for k in ('top','wrist'):
  x=np.load(R/f'initial_states/{a}_images.npz')[k].astype(float);y=np.load(R/f'initial_states/{b}_images.npz')[k].astype(float);out[f'{k}_mae']=float(np.mean(np.abs(x-y)))
 return out
pairs=[('A_ref','B_place'),('B_place','C_transition'),('A_ref','C_transition')]
thr={'q_l2_rad':0.01,'q_max_rad':0.01,'dq_l2':0.1,'ee_pos_mm':1.0,'ee_rot_deg':1.0,'grip_mm':0.5,'obj_pos_mm':0.5,'obj_rot_deg':10.0,'obj_speed_mm_s':1.0,'ee_obj_rel_mm':0.5,'ee_obj_axis_deg':10.0,'contacts_delta':0,'top_mae':1.0,'wrist_mae':1.0}
out=[]
for a,b in pairs:
 d=diff(a,b)
 for k,v in d.items():
  out.append({'pair':f'{a}_vs_{b}','metric':k,'value':v,'threshold':thr[k],'unit':('rad' if k.endswith('_rad') or k=='dq_l2' else 'mm' if k.endswith('_mm') or 'mm_s' in k else 'deg' if k.endswith('_deg') else 'count' if 'contacts' in k else 'pixel'),'pass':bool(v<=thr[k])})
with (D/'state_alignment.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(out[0].keys()));w.writeheader();w.writerows(out)
print('ok')
