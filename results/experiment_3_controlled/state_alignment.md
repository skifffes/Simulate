# State alignment: A_ref vs B_place vs C_transition

State alignment is computed from the complete saved startup states (`initial_states.csv`), images from `CameraRig` 256, and object/EE poses. B/C are the causal comparison; A/B differ by object scene only.

| Variable | A_ref vs B_place | B_place vs C_transition | Threshold | B/C pass |
|---|---:|---:|---:|---|
| Joint q L2 | <1e-7 rad | **8.8e-7 rad** | 0.01 rad | ✅ |
| Joint q max | <1e-7 rad | 5.0e-7 rad | 0.01 rad | ✅ |
| Joint dq L2 | 0 | 0 | 0.1 | ✅ |
| EE position | 3.3e-5 mm | **1.8e-4 mm** | 1 mm | ✅ |
| EE rotation | 0° | 0° | 1° | ✅ |
| Gripper opening | 1.5e-7 mm | 1.6e-9 mm | 0.5 mm | ✅ |
| Object position | **1.864 mm** | **3.6e-8 mm** | 0.5 mm | ✅ |
| Object yaw-axis | 2.54° | 0° | 10° | ✅ |
| Object speed | 2.6e-6 mm/s | 1.4e-10 mm/s | 1 mm/s | ✅ |
| EE–object relative position | **1.864 mm** | 3.6e-8 mm | 0.5 mm | ✅ |
| Contact count | 0 vs 0 | 0 vs 0 | equal | ✅ |
| Top RGB MAE | **0.058** | **0.0** | 1.0 px | ✅ |
| Wrist RGB MAE | 0.0 | **0.0** | 1.0 px | ✅ |

- **B_place and C_transition are state-aligned** on every recorded variable. Their top and wrist images are bit-identical; object and q are identical to numerical precision; velocities are zero; no contacts.
- **A_ref vs B_place** have identical q, q velocity, EE, gripper, wrist image; they differ by **object position 1.864 mm**, yaw-axis 2.54°, and top image MAE 0.058. This is a **scene mismatch**, not a joint-state mismatch.
- **Controller target positions were not saved** in the startup diagnostics; this is a remaining unrecorded variable. The action chunk queue is reset in `run_b`; policy process is fresh per rollout. This limitation is recorded in the report.
- Since A_ref and B_place are both static (dq≈0, object speed≈0, no contacts), velocity/contact differences are not a confound in this pilot.

## Interpretation

- B/C: **State-mismatch: NO** (all recorded thresholds pass). The difference between B and C is the arrival history, not the recorded final state.
- A/B: **State-mismatch: YES for full physical scene** (object position and EE–object relative pose differ by 1.864 mm). A vs B can identify the effect of the object-scene shift, but not object position independently from the corresponding top-image change.
- A/B/C all have the same q and EE and gripper; no group has an object contact or residual velocity.
