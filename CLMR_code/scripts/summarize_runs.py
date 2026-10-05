from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[13,21,42])
    parser.add_argument("--allow-incomplete", action="store_true")
    cli = parser.parse_args()
    values = []
    for seed in cli.seeds:
        folder = cli.run_root / f"seed_{seed}"
        paths = [folder / f"fold_{i:02d}" for i in range(10)] if (folder / "fold_00").exists() else [folder]
        labels, probabilities = [], []
        for run in paths:
            if (run / "test_predictions.npz").exists():
                with np.load(run / "test_predictions.npz", allow_pickle=False) as frame:
                    labels.append(frame["label"]); probabilities.append(frame["probability"])
            elif (run / "test_predictions.csv").exists():
                frame = pd.read_csv(run / "test_predictions.csv")
                labels.append(frame.label.to_numpy()); probabilities.append(frame.probability.to_numpy())
            else:
                if cli.allow_incomplete: break
                raise FileNotFoundError(run)
        if len(labels) != len(paths): continue
        y, p = np.concatenate(labels), np.concatenate(probabilities)
        predicted = (p >= .5).astype(int)
        values.append({"seed":seed, "instances":len(y), "accuracy":float(accuracy_score(y,predicted)),
            "precision":float(precision_score(y,predicted,zero_division=0)), "recall":float(recall_score(y,predicted,zero_division=0)),
            "f1":float(f1_score(y,predicted,zero_division=0))})
    if not values: raise RuntimeError("No complete seed available")
    result = {"per_seed":values, "seeds_completed":[row["seed"] for row in values],
        "aggregate":{key:{"mean":float(np.mean([row[key] for row in values])),
            "std":float(np.std([row[key] for row in values],ddof=1)) if len(values)>1 else None}
            for key in ("accuracy","precision","recall","f1")}}
    (cli.run_root / "summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
