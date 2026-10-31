# State alignment (v2): actual first-inference states

This file uses the **actual state and images captured immediately before the first D2 inference** from the v2 rollouts. It does not use a separate offline 30-step audit.

- **Snapshot:** A/B/C restored from saved state, then one unified 30-step settle.
- **Continuous:** B/C built from the A terminal; C executes the joint transition in the same process, then immediately starts D2; B does the diagnostic qB placement, then starts D2. Both use the same 30-step settle and no intermediate snapshot.

## B/C snapshot

| Variable | Value |
|---|---:|
| q L2 | **9.07×10⁻⁷ rad** |
| q max | 5.26×10⁻⁷ rad |
| dq L2 | 6.55×10⁻¹¹ |
| EE position | **1.79×10⁻⁴ mm** |
| EE rotation | 0° |
| gripper | 1.88×10⁻⁶ mm |
| object position | **3.61×10⁻⁸ mm** |
| object yaw-axis | 1.95×10⁻⁸° |
| **full T_EE⁻¹T_object translation** | **1.91×10⁻⁴ mm** |
| **full T_EE⁻¹T_object rotation** | **0°** |
| top RGB MAE | **0.0** |
| wrist RGB MAE | **0.0** |
| contacts | 0 vs 0 |

## B/C continuous

| Variable | Value |
|---|---:|
| q L2 | **0.0** |
| EE position | **0.0** |
| **relative translation** | **0.0** |
| **relative rotation** | **0.0277°** |
| top/wrist MAE | **0.0** |
| contacts | 0 vs 0 |

Since B and C follow the same qB/grip and settle, their continuous starts are identical at recorded precision. The binary outcomes are also identical for all 5 seeds.

## A/B snapshot

| Variable | Value |
|---|---:|
| q L2 | 5.29×10⁻⁸ rad |
| EE position | 3.33×10⁻⁵ mm |
| EE rotation | 0° |
| object position | **1.864 mm** |
| object yaw-axis | 2.54° |
| **full T_EE⁻¹T_object translation** | **1.864 mm** |
| **full T_EE⁻¹T_object rotation** | **177.46°** |
| top RGB MAE | **0.05798** |
| wrist RGB MAE | 0.0 |

The full relative rotation is 177.46° because the rectangular block is 180°-symmetric; the **axis-equivalent** yaw difference is 2.54°. Both are reported; for contact geometry the axis-equivalent is the physically relevant one, but the full SE(3) value is not hidden.

## Thresholds

| Variable | Threshold | B/C snapshot | B/C continuous | A/B |
|---|---:|---|---|---|
| q L2 | 0.01 rad | PASS | PASS | PASS |
| q max | 0.01 rad | PASS | PASS | PASS |
| dq L2 | 0.1 | PASS | PASS | PASS |
| EE position | 1 mm | PASS | PASS | PASS |
| EE rotation | 1° | PASS | PASS | PASS |
| gripper | 0.5 mm | PASS | PASS | PASS |
| object position | 0.5 mm | PASS | PASS | **FAIL (1.864 mm)** |
| relative translation | 0.5 mm | PASS | PASS | **FAIL (1.864 mm)** |
| relative rotation | 5° | PASS | PASS | **FAIL (177.46°) / axis 2.54°** |
| top RGB MAE | 1 px | PASS | PASS | PASS (0.058) |
| wrist RGB MAE | 1 px | PASS | PASS | PASS |
| contacts | equal | PASS | PASS | PASS |

## Interpretation

- **B/C snapshot:** state-aligned on every recorded variable; first-action q L2 is exactly **0.0**; only later dynamics/numerical amplification can differ.
- **B/C continuous:** state-aligned and **all 5 outcomes identical**; no evidence of a transition-history residual.
- **A/B:** not a pure object-position test; object position and top image are both different. A/B first-action q L2 is 0.0084–0.0103 rad; A and B are both 5/5 in the v2 snapshot run.
- **Controller target:** `q_target` in `initial_state.json` is the actual held q, not an independent commanded target. The actual D2 action target is in `first_action_v2.csv` (`q1..q7`, `grip_mm`). This is an explicit limitation.
- **Observation hashes:** `data/observation_hashes_v2.csv` archives the actual top/wrist SHA-256 prefixes at first inference.
