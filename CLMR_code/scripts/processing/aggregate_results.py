
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRICS = ("accuracy", "precision", "recall", "f1", "macro_f1", "nll", "brier", "ece")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records: list[dict[str, object]] = []
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for path in sorted(args.results.glob("*/*/seed_*/strict_metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = path.parents[2].name
        dataset = path.parents[1].name
        row: dict[str, object] = {
            "model": model,
            "dataset": dataset,
            "seed": int(payload["seed"]),
        }
        test_metrics = payload["test"]
        if "accuracy" not in test_metrics and "acc" in test_metrics:
            test_metrics["accuracy"] = test_metrics["acc"]
        row.update({metric: float(test_metrics[metric]) for metric in METRICS})
        records.append(row)
        grouped.setdefault((model, dataset), []).append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_path = args.output.with_name(f"{args.output.stem}_raw.csv")
    with raw_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("model", "dataset", "seed", *METRICS))
        writer.writeheader()
        writer.writerows(records)

    summary_rows: list[dict[str, object]] = []
    for (model, dataset), values in sorted(grouped.items()):
        row: dict[str, object] = {
            "model": model,
            "dataset": dataset,
            "seeds": len(values),
        }
        for metric in METRICS:
            samples = np.asarray([float(value[metric]) for value in values])
            row[f"{metric}_mean"] = float(samples.mean())
            row[f"{metric}_std"] = float(samples.std(ddof=1)) if len(samples) > 1 else ""
        summary_rows.append(row)

    fields = ["model", "dataset", "seeds"]
    for metric in METRICS:
        fields.extend((f"{metric}_mean", f"{metric}_std"))
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"runs={len(records)} groups={len(summary_rows)}")
    print(args.output)


if __name__ == "__main__":
    main()
