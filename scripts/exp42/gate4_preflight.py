#!/usr/bin/env python3
"""Exp 4.2 Gate-4 preflight: audit representation semantics and contact timing.

Read-only, CPU-only, NumPy + standard library; no OpenPI/PyBullet/GPU required.

Train-data action representations, all for the SAME expert target sequence A:
  anchor[k] = A[t+k] - S[t]
  observed_residual[k] = A[t+k] - S[t+k]
  target_increment[0] = A[t] - S[t]
  target_increment[k>0] = A[t+k] - A[t+k-1]

Only `anchor` with FIXED interpretation and `target_increment` with RECURSIVE
interpretation reconstruct all future expert targets exactly. Recursively
summing `observed_residual` generally does not, due to tracking lag.

Usage:
  python exp42_gate4_preflight.py dataset \
    --data /scratch/lijh/exp42/data \
    --stats /path/to/assets/droid/norm_stats.json \
    --out gate4_dataset_preflight.json

  python exp42_gate4_preflight.py rollout \
    --csv results/experiment_4_2/rollouts_p0_logged/B90_ep00_substeps.csv \
    --out gate4_contact_timeline.json

  python exp42_gate4_preflight.py self-test
"""
from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path

import numpy as np


def forms(states: np.ndarray, targets: np.ndarray, t: int, horizon: int = 16):
    s = np.asarray(states, dtype=float)
    a = np.asarray(targets, dtype=float)
    if s.ndim != 2 or a.ndim != 2 or len(s) != len(a):
        raise ValueError("states/targets must be same-length 2D arrays")
    if s.shape[1] < 9 or a.shape[1] < 8:
        raise ValueError(f"Need state>=9, action>=8 dimensions; got {s.shape}, {a.shape}")
    if t < 0 or t + horizon > len(s):
        raise ValueError("Window out of range; preflight intentionally excludes padded windows")
    q0 = s[t, :7]
    q_obs = s[t:t + horizon, :7]
    q_targets = a[t:t + horizon, :7]
    anchor = q_targets - q0
    observed_residual = q_targets - q_obs
    inc = np.empty_like(q_targets)
    inc[0] = q_targets[0] - q0
    inc[1:] = q_targets[1:] - q_targets[:-1]
    gripper = np.clip(a[t:t + horizon, 7] / .04, 0, 1)
    return {
        "anchor": anchor,
        "observed_residual": observed_residual,
        "target_increment": inc,
        "q0": q0,
        "q_targets": q_targets,
        "gripper_sequence": gripper,
    }


def reconstruction_errors(f):
    reference = f["q_targets"]
    q0 = f["q0"]
    anchor_recon = q0 + f["anchor"]
    residual_recursive = q0 + np.cumsum(f["observed_residual"], axis=0)
    increment_recursive = q0 + np.cumsum(f["target_increment"], axis=0)
    return {
        "anchor_fixed_max": float(np.max(np.abs(anchor_recon - reference))),
        "observed_residual_recursive_max": float(
            np.max(np.abs(residual_recursive - reference))
        ),
        "observed_residual_recursive_final_max": float(
            np.max(np.abs(residual_recursive[-1] - reference[-1]))
        ),
        "target_increment_recursive_max": float(
            np.max(np.abs(increment_recursive - reference))
        ),
    }


def find_action_stats(d):
    if isinstance(d, dict):
        for key in ("actions", "action"):
            v = d.get(key)
            if isinstance(v, dict) and "q01" in v and "q99" in v:
                return (
                    np.asarray(v["q01"], dtype=float)[:7],
                    np.asarray(v["q99"], dtype=float)[:7],
                )
        for value in d.values():
            try:
                return find_action_stats(value)
            except ValueError:
                pass
    raise ValueError("Cannot find action q01/q99 stats")


def distribution(x):
    x = np.asarray(x, dtype=float)
    return {
        "q01_by_joint": np.quantile(x, .01, axis=(0, 1)).tolist(),
        "q99_by_joint": np.quantile(x, .99, axis=(0, 1)).tolist(),
        "abs_mean_by_horizon_row": np.abs(x).mean(axis=(0, 2)).tolist(),
        "abs_q95": float(np.quantile(np.abs(x), .95)),
    }


def analyze_dataset(data: Path, stats: Path | None = None,
                    horizon: int = 16, stride: int = 4):
    metadata_path = data / "meta.jsonl"
    metadata = [
        json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    coll = {"anchor": [], "observed_residual": [], "target_increment": []}
    err_list, grip_change = [], []
    episodes = 0
    for row in metadata:
        if row.get("split") != "train":
            continue
        episodes += 1
        npz_path = data / row["state_path"]
        with np.load(npz_path, allow_pickle=False) as z:
            s, a = np.asarray(z["state"]), np.asarray(z["action"])
        if len(s) < horizon:
            raise ValueError(f"Episode shorter than horizon: {npz_path}")
        for t in range(0, len(s) - horizon + 1, stride):
            f = forms(s, a, t, horizon)
            for name in coll:
                coll[name].append(f[name])
            err_list.append(reconstruction_errors(f))
            g = f["gripper_sequence"]
            grip_change.append(bool(np.any(np.abs(g[1:] - g[0]) > 1e-9)))
    if not err_list:
        raise ValueError("No train windows")
    result = {
        "scope": "train split ONLY; read-only; no padding windows",
        "episodes": episodes, "windows": len(err_list),
        "horizon": horizon, "stride": stride,
        "definitions": {
            "anchor": "A[t+k] - S[t]; execute FIXED",
            "observed_residual": "A[t+k] - S[t+k]; recursively summing is NOT expert equivalent",
            "target_increment": "k=0: A[t]-S[t], k>0: A[t+k]-A[t+k-1]; execute RECURSIVE",
            "gripper": "A[t+k,7] / 0.04 at EACH future row; not a constant t value",
        },
        "reconstruction_errors": {
            key: {
                "max_over_windows": max(e[key] for e in err_list),
                "mean_over_windows": float(np.mean([e[key] for e in err_list])),
                "p95_over_windows": float(np.percentile([e[key] for e in err_list], 95)),
            } for key in err_list[0]
        },
        "windows_with_future_gripper_change": int(sum(grip_change)),
        "fraction_windows_with_future_gripper_change": float(np.mean(grip_change)),
        "distributions": {},
    }
    for name, vals in coll.items():
        x = np.stack(vals)
        result["distributions"][name] = distribution(x)
    if stats is not None:
        q01, q99 = find_action_stats(json.loads(stats.read_text(encoding="utf-8")))
        delta = np.maximum(q99 - q01, 1e-9)
        for name, vals in coll.items():
            x = np.stack(vals)
            info = result["distributions"][name]
            lo = np.quantile(x, .01, axis=(0, 1))
            hi = np.quantile(x, .99, axis=(0, 1))
            normed = 2 * (x - q01) / delta - 1
            info["qspan_over_droid_by_joint"] = ((hi - lo) / delta).tolist()
            info["normalized_outside_minus1_plus1_fraction"] = float(
                (np.abs(normed) > 1).mean()
            )
        result["droid_stats_source"] = str(stats)
    return result


def vec(s):
    return np.fromstring(s, dtype=float, sep=";")


def analyze_rollout(csv_path: Path):
    with csv_path.open(encoding="utf-8", newline="") as f:
        records = list(csv.DictReader(f))
    if not records:
        raise ValueError("No rows in rollout CSV")
    required = {"tick", "k", "q_target", "q_before", "q_after",
                "robot_block_contacts", "block_wall_contacts",
                "ee_dx", "ee_dy", "obj_dx", "obj_dy"}
    missing = required.difference(records[0])
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    touches = [
        int(float(r["robot_block_contacts"])) > 0 or
        int(float(r["block_wall_contacts"])) > 0
        for r in records
    ]
    first = next((i for i, v in enumerate(touches) if v), None)
    segments = []
    start = None
    for i, touch in enumerate(touches + [False]):
        if touch and start is None:
            start = i
        if not touch and start is not None:
            segments.append({
                "start_step": start, "end_step": i - 1,
                "steps": i - start,
                "start_tick_k": [int(records[start]["tick"]), int(records[start]["k"])],
                "end_tick_k": [int(records[i - 1]["tick"]), int(records[i - 1]["k"])],
            })
            start = None

    first_tick = int(records[first]["tick"]) if first is not None else None
    first_post_contact_replan = next(
        ({"index": i, "tick": int(r["tick"]), "k": int(r["k"])}
         for i, r in enumerate(records)
         if first_tick is not None
         and int(r["tick"]) > first_tick and int(r["k"]) == 0),
        None,
    )
    # These are *joint-command target path* increments; no FK is approximated.
    target_deltas, actual_deltas = [], []
    prev_q_target = vec(records[0]["q_before"])
    for r in records:
        target_q = vec(r["q_target"])
        before_q = vec(r["q_before"])
        after_q = vec(r["q_after"])
        target_deltas.append(target_q - prev_q_target)
        actual_deltas.append(after_q - before_q)
        prev_q_target = target_q
    target_deltas, actual_deltas = np.asarray(target_deltas), np.asarray(actual_deltas)

    def aggregate(indices):
        if not indices:
            return None
        return {
            "steps": len(indices),
            "actual_ee_dx": sum(float(records[i]["ee_dx"]) for i in indices),
            "actual_ee_dy": sum(float(records[i]["ee_dy"]) for i in indices),
            "object_dx": sum(float(records[i]["obj_dx"]) for i in indices),
            "object_dy": sum(float(records[i]["obj_dy"]) for i in indices),
            "mean_abs_joint_command_step_rad": float(np.abs(target_deltas[indices]).mean()),
            "mean_abs_joint_actual_step_rad": float(np.abs(actual_deltas[indices]).mean()),
            "max_joint_command_step_rad": float(np.abs(target_deltas[indices]).max()),
        }
    pre = [i for i in range(len(records)) if first is None or i < first]
    active = [i for i, v in enumerate(touches) if v]
    after_not_touch = [i for i in range(len(records))
                       if first is not None and i >= first and not touches[i]]
    replan_after = [i for i, r in enumerate(records)
                    if first_tick is not None and int(r["tick"]) > first_tick]
    return {
        "source": str(csv_path),
        "n_substeps": len(records),
        "first_contact": None if first is None else {
            "index": first, "tick": first_tick, "k": int(records[first]["k"]),
            "kind": "robot_block and/or block_wall (end-of-substep measurement)",
        },
        "first_policy_replan_after_first_contact": first_post_contact_replan,
        "contact_bursts": segments,
        "n_contact_bursts": len(segments),
        "n_contact_substeps": len(active),
        "contact_substeps_with_k_gt_zero": sum(int(records[i]["k"]) > 0 for i in active),
        "substep_groups": {
            "strictly_before_first_contact": aggregate(pre),
            "contact_flag_active_not_necessarily_contiguous": aggregate(active),
            "noncontact_after_first_contact": aggregate(after_not_touch),
            "all_steps_after_first_contact_tick": aggregate(replan_after),
        },
        "important_limitations": [
            "Contact is measured AFTER executing the substep; its action was planned at tick start.",
            "The first policy chunk containing contact may have been generated before any contact.",
            "Contact-flagged substeps need not be a contiguous temporal phase.",
            "This file does not estimate command EE trajectory; FK(qtarget) per step is required.",
            "Summing FK(qtarget_i)-FK(qbefore_i) is summing target errors, NOT EE command-path travel.",
        ],
    }


def self_test():
    s = np.zeros((20, 9), dtype=float)
    a = np.zeros((20, 8), dtype=float)
    a[:, :7] = np.arange(20)[:, None] * .04
    s[:, :7] = np.maximum(np.arange(20) - 1, 0)[:, None] * .025
    a[:, 7] = .04
    a[6:, 7] = .017
    f = forms(s, a, 0, 16)
    err = reconstruction_errors(f)
    assert err["anchor_fixed_max"] < 1e-12
    assert err["target_increment_recursive_max"] < 1e-12
    assert err["observed_residual_recursive_final_max"] > .1
    assert f["gripper_sequence"][0] == 1
    assert abs(f["gripper_sequence"][6] - .425) < 1e-10

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "B90").mkdir()
        np.savez(d / "B90" / "ep.npz", state=s, action=a)
        (d / "meta.jsonl").write_text(json.dumps({
            "split": "train", "skill": "B90", "state_path": "B90/ep.npz"
        }) + "\n")
        rep = analyze_dataset(d)
        assert rep["episodes"] == 1 and rep["windows"] == 2
        assert rep["windows_with_future_gripper_change"] == 2
        assert rep["reconstruction_errors"]["target_increment_recursive_max"]["max_over_windows"] < 1e-12
        path = d / "events.csv"
        rows = []
        # First contact is tick=1,k=2, disappears before next model inference.
        for tick in range(3):
            for k in range(4):
                idx = 4 * tick + k
                rows.append({
                    "tick": tick, "k": k,
                    "q_target": ";".join([f"{idx * .01}"] * 7),
                    "q_before": ";".join([f"{max(idx - 1, 0) * .01}"] * 7),
                    "q_after": ";".join([f"{idx * .01}"] * 7),
                    "robot_block_contacts": 1 if idx in (6, 7, 10) else 0,
                    "block_wall_contacts": 0,
                    "ee_dx": .01, "ee_dy": 0,
                    "obj_dx": 0, "obj_dy": 0,
                })
        with path.open("w", encoding="utf-8", newline="") as fcsv:
            writer = csv.DictWriter(fcsv, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        event = analyze_rollout(path)
        assert event["first_contact"]["tick"] == 1
        assert event["first_contact"]["k"] == 2
        assert event["first_policy_replan_after_first_contact"]["tick"] == 2
        assert event["n_contact_bursts"] == 2
    print("PASS: three action representations, future gripper, and contact/event alignment")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    cmd = parser.add_subparsers(dest="mode", required=True)
    p_data = cmd.add_parser("dataset")
    p_data.add_argument("--data", required=True, type=Path)
    p_data.add_argument("--stats", type=Path)
    p_data.add_argument("--horizon", type=int, default=16)
    p_data.add_argument("--stride", type=int, default=4)
    p_data.add_argument("--out", type=Path, required=True)
    p_rollout = cmd.add_parser("rollout")
    p_rollout.add_argument("--csv", required=True, type=Path)
    p_rollout.add_argument("--out", required=True, type=Path)
    cmd.add_parser("self-test")
    args = parser.parse_args()
    if args.mode == "self-test":
        self_test()
        return
    if args.mode == "dataset":
        output = analyze_dataset(args.data, args.stats, args.horizon, args.stride)
    else:
        output = analyze_rollout(args.csv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print("Wrote", args.out)
    if args.mode == "dataset":
        for name, item in output["reconstruction_errors"].items():
            print(f"  {name}: max={item['max_over_windows']:.8f} rad")
        print("  windows with future gripper change:",
              output["windows_with_future_gripper_change"])
    else:
        print("  first contact:", output["first_contact"])
        print("  contact bursts:", output["n_contact_bursts"])
        print("  first subsequent replan:",
              output["first_policy_replan_after_first_contact"])


if __name__ == "__main__":
    main()
