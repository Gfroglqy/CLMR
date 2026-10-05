
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def canonical(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "sentence": frame["sentence"],
        "label": frame["label"].astype(int),
        "target_position": frame["w_index"].astype(int),
        "target_word": frame["target"],
        "pos_tag": frame["POS"],
        "gloss": frame["definition"].fillna(frame["target"]),
        "eg_sent": frame["sentence"],
        "source_id": frame["index"],
    })


def melbert(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[["index", "label", "sentence", "POS", "FGPOS", "w_index"]].copy()


def stats(frame: pd.DataFrame) -> dict[str, float | int]:
    return {
        "targets": int(len(frame)),
        "sentences": int(frame["sentence"].nunique()),
        "positive_rate": float(frame["label"].astype(int).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--validation-fraction", type=float, default=0.0416)
    args = parser.parse_args()

    full_train = pd.read_csv(args.source / "train.tsv", sep="\t")
    test = pd.read_csv(args.source / "test.tsv", sep="\t")
    if len(full_train) != 160_154 or len(test) != 22_196:
        raise ValueError(f"Unexpected VUA-20 counts: {len(full_train)}/{len(test)}")

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=args.validation_fraction, random_state=args.split_seed
    )
    train_idx, val_idx = next(splitter.split(full_train, groups=full_train["sentence"]))
    train = full_train.iloc[train_idx].reset_index(drop=True)
    val = full_train.iloc[val_idx].reset_index(drop=True)
    if set(train["sentence"]) & set(val["sentence"]):
        raise RuntimeError("Sentence leakage detected between VUA-20 train and validation")

    canonical_root = args.output_root / "canonical"
    melbert_root = args.output_root / "melbert"
    canonical_root.mkdir(parents=True, exist_ok=True)
    melbert_root.mkdir(parents=True, exist_ok=True)
    for name, frame in (("train", train), ("val", val), ("test", test)):
        canonical(frame).to_csv(canonical_root / f"{name}.csv", index=False)
        melbert(frame).to_csv(
            melbert_root / f"{'dev' if name == 'val' else name}.tsv",
            sep="\t", index=False,
        )

    manifest = {
        "dataset": "VUA-20 strict local extension",
        "source_train_targets": int(len(full_train)),
        "source_test_targets": int(len(test)),
        "split_seed": args.split_seed,
        "validation_fraction": args.validation_fraction,
        "splits": {name: stats(frame) for name, frame in (("train", train), ("validation", val), ("test", test))},
        "protocol_note": (
            "Released test is unchanged. Validation is a fixed sentence-disjoint subset "
            "removed from released training because the local release has no dev.tsv."
        ),
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
