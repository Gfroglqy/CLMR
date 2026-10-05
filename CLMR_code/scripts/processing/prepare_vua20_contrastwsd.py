
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


KEYS = ["index", "label", "sentence", "w_index"]


def canonical_keys(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return frame.rename(
        columns={"source_id": "index", "target_position": "w_index"}
    )[["index", "label", "sentence", "w_index"]]


def exact_select(official: pd.DataFrame, selected: pd.DataFrame, split: str) -> pd.DataFrame:
    official = official.copy()
    selected = selected.copy()
    for frame in (official, selected):
        frame["label"] = frame["label"].astype(int)
        frame["w_index"] = frame["w_index"].astype(int)
        frame["index"] = frame["index"].astype(str)

    if official.duplicated(KEYS).any() or selected.duplicated(KEYS).any():
        raise ValueError(f"Non-unique VUA-20 alignment key in {split}")
    aligned = selected.merge(
        official, on=KEYS, how="left", validate="one_to_one", indicator=True
    )
    missing = int((aligned["_merge"] != "both").sum())
    if missing:
        raise ValueError(f"ContrastWSD alignment missed {missing} rows in {split}")


    for column in ("target", "word_sense", "definition"):
        if column in aligned:
            aligned[column] = aligned[column].fillna(aligned.get("target", ""))
    return aligned[official.columns]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    official_train = pd.read_csv(args.official_root / "train.tsv", sep="\t")
    official_test = pd.read_csv(args.official_root / "test.tsv", sep="\t")
    splits = {
        "train": exact_select(
            official_train, canonical_keys(args.canonical_root / "train.csv"), "train"
        ),
        "dev": exact_select(
            official_train, canonical_keys(args.canonical_root / "val.csv"), "dev"
        ),
        "test": exact_select(
            official_test, canonical_keys(args.canonical_root / "test.csv"), "test"
        ),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.to_csv(args.output_root / f"{name}.tsv", sep="\t", index=False)

    manifest = {
        "dataset": "VUA-20 ContrastWSD aligned",
        "alignment_keys": KEYS,
        "splits": {name: int(len(frame)) for name, frame in splits.items()},
        "protocol_note": (
            "Released ContrastWSD features; exact row selection follows the fixed "
            "sentence-disjoint VUA-20 local development split."
        ),
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
