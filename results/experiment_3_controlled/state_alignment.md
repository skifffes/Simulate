# State alignment (v4): actual first-inference states

This file uses the **actual state and images captured immediately before the first D2 inference**. v4 adds a target-convergence gate and verifies actual gripper state and the real model input tensors.

## V4 continuous B/C

| Variable | Value |
|---|---:|
| q error to qB | **5.20×10⁻⁶ rad** |
| q max error | ≤5.20×10⁻⁶ rad |
| dq L2 | **5.68×10⁻¹³** |
| convergence steps | **201** (0.8375 s) |
| EE position difference B/C | **0.0 mm** |
| dq difference B/C | **0.0** |
| object position / velocity difference | **0.0 / 0.0** |
| relative translation B/C | **0.0 mm** |
| relative rotation B/C | **0.0°** |
| **actual left finger** | **35.00026 mm** |
| **actual right finger** | **34.99980 mm** |
| **actual gripper B/C difference** | **0.0 mm** |
| **startup contacts** | **0** |
| **9D model state input L2** | **0.0** |
| **top/wrist normalized input MAE** | **0.0 / 0.0** |
| top RGB hash | **equal** |
| wrist RGB hash | **equal** |
| first action 7-joint L2 | **0.0** |
| success | 4/5 both; same failure seed4 |

Convergence criteria: q L2≤0.01 rad, q max≤0.01, dq L2≤0.2, object speed≤0.01 m/s, grip error≤2 mm, held 120 steps, max 1200 steps. All 10 runs converged in 201 steps. The pre-v4 run had q error 0.671 rad and EE error 207 mm; it is invalid and replaced. V4 physically adjusts the gripper from the A-open state with `set_gripper` (no `resetJointState` for fingers); the B/C actual finger positions are equal at 35.00026/34.99980 mm.

## Snapshot B/C (v2 state audit; still valid)

| Variable | Value |
|---|---:|
| q L2 | **9.07×10⁻⁷ rad** |
| EE position | **1.79×10⁻⁴ mm** |
| object position | **3.61×10⁻⁸ mm** |
| relative translation | **1.91×10⁻⁴ mm** |
| relative rotation | **0°** |
| top/wrist MAE | **0 / 0** |
| first action 7-joint L2 | **0.0** |
| contacts | 0 vs 0 |

## A/B snapshot

| Variable | Value |
|---|---:|
| q L2 | 5.29×10⁻⁸ rad |
| object position | **1.864 mm** |
| **relative translation** | **1.864 mm** |
| **relative rotation** | **177.46°** |
| axis-equivalent yaw | **2.54°** |
| top MAE | **0.05798** |
| wrist MAE | 0.0 |
| first action 7-joint L2 | **0.0084–0.0103 rad** |

The rectangular block is 180°-symmetric; full relative quaternion rotation is 177.46°, axis-equivalent yaw difference is 2.54°. Both are reported.

## Thresholds

| Variable | Threshold | B/C snapshot | B/C continuous v3 | A/B |
|---|---:|---|---|---|
| q L2 | 0.01 rad | PASS | PASS (5.2e-6 to qB) | PASS |
| q max | 0.01 rad | PASS | PASS | PASS |
| dq L2 | 0.1 | PASS | PASS | PASS |
| EE position | 1 mm | PASS | PASS | PASS |
| EE rotation | 1° | PASS | PASS | PASS |
| gripper | 0.5 mm | PASS | PASS | PASS |
| object position | 0.5 mm | PASS | PASS | **FAIL (1.864 mm)** |
| relative translation | 0.5 mm | PASS | PASS | **FAIL (1.864 mm)** |
| relative rotation | 5° | PASS | PASS | **FAIL (177.46° full) / axis 2.54°** |
| top RGB MAE | 1 px | PASS | PASS (hash equal) | PASS (0.058) |
| wrist RGB MAE | 1 px | PASS | PASS (hash equal) | PASS |
| contacts | equal | PASS | PASS | PASS |

## Interpretation

- **B/C continuous v3:** reaches qB, is state-aligned at float precision, has identical first action and identical 5-seed outcomes. No evidence of a transition-history residual.
- **B/C snapshot:** state-aligned, first action identical, but C seed4 fails while B seed4 succeeds; binary success is sensitive to later dynamics/numerics and is not a state-equivalence test.
- **A/B:** not a pure object-position test: object position and top image differ together.
- **Controller target:** v3 `initial_state.json` has `q_target=qB`; raw full-precision first actions are in `first_action_v3.csv`.
- **Observation hashes:** `data/observation_hashes_v3.csv`.


