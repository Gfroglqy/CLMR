"""Create leakage-safe outer-10-fold data for canonical TroFi and MOH-X.

For each outer test fold, a stratified development subset is drawn only from
the remaining 90% training examples.  The dev split selects the checkpoint;
the held-out outer fold is never used for epoch or router selection.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


FIELDS = ["sentence", "label", "target_position", "target_word", "pos_tag", "gloss", "eg_sent"]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {path}")
    for row in rows:
        if not set(FIELDS).issubset(row):
            raise ValueError(f"Unexpected schema in {path}: {row.keys()}")
        tokens = row["sentence"].split()
        if not 0 <= int(row["target_position"]) < len(tokens):
            raise ValueError(f"Invalid target index: {row}")
    return rows


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()
    rows = read_rows(args.source)
    labels = [int(row["label"]) for row in rows]
    outer = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    manifest = {"source": str(args.source.resolve()), "rows": len(rows), "folds": args.folds,
                "seed": args.seed, "development_fraction_of_outer_train": 1 / 9, "splits": []}
    for fold, (outer_train, outer_test) in enumerate(outer.split(range(len(rows)), labels)):
        outer_train_labels = [labels[index] for index in outer_train]
        inner = StratifiedShuffleSplit(n_splits=1, test_size=1 / 9, random_state=args.seed + fold)
        train_local, dev_local = next(inner.split(outer_train, outer_train_labels))
        train_indices = [outer_train[index] for index in train_local]
        dev_indices = [outer_train[index] for index in dev_local]
        fold_dir = args.output / f"fold_{fold:02d}"
        write_rows(fold_dir / "train.csv", [rows[index] for index in train_indices])
        write_rows(fold_dir / "val.csv", [rows[index] for index in dev_indices])
        write_rows(fold_dir / "test.csv", [rows[index] for index in outer_test])
        manifest["splits"].append({"fold": fold, "train": len(train_indices), "val": len(dev_indices),
                                   "test": len(outer_test), "test_positive_rate":
                                   sum(labels[index] for index in outer_test) / len(outer_test)})
    import json
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(manifest)


if __name__ == "__main__":
    main()
