
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = ("accuracy", "precision", "recall", "f1", "macro_f1", "nll", "brier", "ece")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    values = []
    for seed in (13, 21, 42):
        path = args.dataset_root / f"seed_{seed}" / "crossval_metrics.json"
        if not path.exists():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        values.append({"seed": seed, **payload})
    summary: dict[str, object] = {"seeds": [13, 21, 42], "per_seed": values}
    for metric in METRICS:
        samples = np.asarray([float(item[metric]) for item in values])
        summary[f"{metric}_mean"] = float(samples.mean())
        summary[f"{metric}_std"] = float(samples.std(ddof=1))
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
