#!/usr/bin/env python3
"""**导入自外部审计**（`audit_action_dataset.py`，2026-11-04）——只读审计训练动作标签。

用途：在真实 `/scratch/lijh/exp42/data` 上重建 `train_lora.make_item` 的 16 步固定锚点标签，
给出逐 k、逐关节的统计与 DROID 归一化空间跨度，用于判断「归一化是否把有效信号压得太窄」。

保留审计方原始实现（仅在头部加来源说明），只读、无模型、无 GPU。

用法：
  python scripts/exp42/audit_action_dataset.py --data /scratch/lijh/exp42/data \
      --out results/experiment_4_2/contract_replay_p0c/action_contract_audit.json \
      --stats /scratch/lijh/.cache/openpi/openpi-assets/checkpoints/pi05_droid/assets/droid/norm_stats.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

HORIZON = 16
STRIDE = 4
JOINTS = 7


def stats_summary(x):
    x = np.asarray(x, np.float64)
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "q01": float(np.quantile(x, 0.01)),
        "median": float(np.median(x)),
        "q99": float(np.quantile(x, 0.99)),
        "abs_q95": float(np.quantile(np.abs(x), 0.95)),
        "absmax": float(np.max(np.abs(x))),
    }


def find_action_quantiles(obj):
    if isinstance(obj, dict):
        for key in ("actions", "action"):
            v = obj.get(key)
            if isinstance(v, dict) and "q01" in v and "q99" in v:
                return np.asarray(v["q01"], float)[:JOINTS], np.asarray(v["q99"], float)[:JOINTS]
        for v in obj.values():
            try:
                return find_action_quantiles(v)
            except ValueError:
                continue
    raise ValueError("Cannot find action q01/q99 in norm_stats.json")


def collect(data_root):
    path = data_root / "meta.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing: {path}")
    meta = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    per_skill = {}
    all_step, all_anchor = [], []
    n_eps = n_windows = 0
    for row in meta:
        if row.get("split") != "train":
            continue
        n_eps += 1
        sa = np.load(data_root / row["state_path"], allow_pickle=False)
        S = np.asarray(sa["state"], np.float64)
        A = np.asarray(sa["action"], np.float64)
        if S.ndim != 2 or A.ndim != 2 or S.shape[1] < 7 or A.shape[1] < 8:
            raise ValueError(f"unexpected dimensions in {row['state_path']}: {S.shape}, {A.shape}")
        if len(S) != len(A):
            raise ValueError(f"state/action length mismatch in {row['state_path']}")
        if len(S) < HORIZON:
            raise ValueError(f"short episode {row['state_path']}: {len(S)} < {HORIZON}")
        step_windows, anchor_windows = [], []
        for t in range(0, len(S) - HORIZON + 1, STRIDE):
            tgt = A[t:t + HORIZON, :JOINTS]
            step_windows.append(tgt - S[t:t + HORIZON, :JOINTS])
            anchor_windows.append(tgt - S[t, :JOINTS][None, :])
        sw = np.stack(step_windows)
        aw = np.stack(anchor_windows)
        n_windows += len(sw)
        all_step.append(sw)
        all_anchor.append(aw)
        skill = row.get("skill", "unknown")
        per_skill.setdefault(skill, [[], []])
        per_skill[skill][0].append(sw)
        per_skill[skill][1].append(aw)
    if not all_step:
        raise ValueError("No TRAIN episodes in meta.jsonl")
    step = np.concatenate(all_step)
    anchored = np.concatenate(all_anchor)
    return meta, n_eps, n_windows, step, anchored, per_skill


def audit(data_root, stats_path=None):
    _, n_eps, n_windows, step, anchored, by_skill = collect(data_root)
    result = {
        "scope": "train split only; metadata and arrays read-only",
        "contract": "step[k]=A[t+k]-S[t+k]; anchored[k]=A[t+k]-S[t] (actual train_lora.make_item)",
        "horizon": HORIZON,
        "stride": STRIDE,
        "train_episodes": n_eps,
        "train_windows": n_windows,
        "step_joint": [stats_summary(step[:, :, j]) for j in range(JOINTS)],
        "anchored_joint": [stats_summary(anchored[:, :, j]) for j in range(JOINTS)],
        "by_row": [
            {
                "k": k,
                "step_abs_mean": float(np.abs(step[:, k, :]).mean()),
                "anchored_abs_mean": float(np.abs(anchored[:, k, :]).mean()),
                "anchored_abs_q95": float(np.quantile(np.abs(anchored[:, k, :]), .95)),
                "max_abs_diff_anchor_vs_step": float(np.abs(anchored[:, k, :]
                                                           - step[:, k, :]).max()),
            } for k in range(HORIZON)
        ],
        "per_skill": {},
    }
    for skill, (sw, aw) in by_skill.items():
        s = np.concatenate(sw)
        a = np.concatenate(aw)
        result["per_skill"][skill] = {
            "windows": int(len(s)),
            "step_joint_q01": np.quantile(s, .01, axis=(0, 1)).tolist(),
            "step_joint_q99": np.quantile(s, .99, axis=(0, 1)).tolist(),
            "anchored_joint_q01": np.quantile(a, .01, axis=(0, 1)).tolist(),
            "anchored_joint_q99": np.quantile(a, .99, axis=(0, 1)).tolist(),
        }
    if stats_path is not None:
        j = json.loads(stats_path.read_text(encoding="utf-8"))
        q01, q99 = find_action_quantiles(j)
        denom = q99 - q01 + 1e-6
        norm_step = 2.0 * (step - q01) / denom - 1.0
        norm_anchor = 2.0 * (anchored - q01) / denom - 1.0
        round_trip = (norm_anchor + 1.0) * denom / 2.0 + q01
        result["droid_norm"] = {
            "file": str(stats_path),
            "action_joint_q01": q01.tolist(),
            "action_joint_q99": q99.tolist(),
            "qspan_train_anchor_over_droid": [
                float((np.quantile(anchored[:, :, k], .99) - np.quantile(anchored[:, :, k], .01))
                      / max(q99[k] - q01[k], 1e-9)) for k in range(JOINTS)
            ],
            "normalized_anchored": [stats_summary(norm_anchor[:, :, k]) for k in range(JOINTS)],
            "normalized_step": [stats_summary(norm_step[:, :, k]) for k in range(JOINTS)],
            "anchored_frac_outside_minus1_plus1": float((np.abs(norm_anchor) > 1).mean()),
            "round_trip_max_abs_err": float(np.max(np.abs(round_trip - anchored))),
        }
    assert np.allclose(step[:, 0, :], anchored[:, 0, :], atol=1e-12)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stats", type=Path, default=None)
    args = ap.parse_args()
    result = audit(args.data, args.stats)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TRAIN episodes={result['train_episodes']} windows={result['train_windows']}")
    print("row k | step_abs_mean | anchored_abs_mean | anchored_abs_q95")
    for row in result["by_row"]:
        print(f"{row['k']:>5} | {row['step_abs_mean']:.6f} | {row['anchored_abs_mean']:.6f} | "
              f"{row['anchored_abs_q95']:.6f}")
    if "droid_norm" in result:
        print("anchored action qspan / DROID:", result["droid_norm"]["qspan_train_anchor_over_droid"])
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
