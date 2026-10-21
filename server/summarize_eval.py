"""汇总一个或多个 eval.json 的关键指标，便于对比不同 checkpoint / 种子集合。"""
import json
import sys
from pathlib import Path

KEYS = ("policy", "episodes", "success", "success_rate", "failure_counts",
        "mean_lift_height_m", "mean_place_xy_error_m", "mean_ticks")


def main(paths):
    for path in paths:
        path = Path(path)
        if not path.exists():
            print(f"{path}: 尚未生成")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        summary = data["summary"]
        row = {key: summary.get(key) for key in KEYS}
        print(path.parent.name, json.dumps(row, ensure_ascii=False))
        failed = [ep["seed"] for ep in data["episodes"] if not ep["success"]]
        if failed:
            print("  failed seeds:", failed)


if __name__ == "__main__":
    main(sys.argv[1:])
