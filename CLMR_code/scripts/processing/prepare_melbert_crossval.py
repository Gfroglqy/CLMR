
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


HEADER = ["index", "label", "sentence", "POS", "FGPOS", "w_index"]
SOURCE_FILES = {"train": "train.csv", "dev": "val.csv", "test": "test.csv"}


def convert_split(source: Path, output: Path, prefix: str) -> dict[str, int | float]:
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty split: {source}")

    positives = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(HEADER)
        for row_number, row in enumerate(rows):
            sentence = row["sentence"]
            target_index = int(row["target_position"])
            tokens = sentence.split()
            if not 0 <= target_index < len(tokens):
                raise ValueError(f"Invalid target index in {source}: {row}")
            label = int(row["label"])
            if label not in (0, 1):
                raise ValueError(f"Invalid label in {source}: {row}")
            positives += label
            pos = row.get("pos_tag", "VERB") or "VERB"
            writer.writerow(
                [f"{prefix}-{row_number}", label, sentence, pos, pos, target_index]
            )
    return {"instances": len(rows), "positive_rate": positives / len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("mohx", "trofi"), required=True)
    args = parser.parse_args()

    manifest: dict[str, object] = {
        "dataset": args.dataset,
        "source_root": str(args.source_root.resolve()),
        "folds": [],
    }
    for fold in range(10):
        source_fold = args.source_root / args.dataset / f"fold_{fold:02d}"
        output_fold = args.output_root / args.dataset / f"fold_{fold:02d}"
        fold_stats: dict[str, object] = {"fold": fold}
        for split, filename in SOURCE_FILES.items():
            fold_stats[split] = convert_split(
                source_fold / filename,
                output_fold / f"{split}.tsv",
                f"{args.dataset}-f{fold:02d}-{split}",
            )
        manifest["folds"].append(fold_stats)

    destination = args.output_root / args.dataset / "manifest.json"
    destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
