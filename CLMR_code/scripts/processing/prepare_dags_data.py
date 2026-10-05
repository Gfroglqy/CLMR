
from __future__ import annotations

import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path

import spacy
from spacy.tokens import Doc


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return [
        {
            "group": f'{row["index"]}\t{row["sentence"]}',
            "sentence": row["sentence"],
            "label": row["label"],
            "target_position": row["w_index"],
            "pos_tag": row.get("POS", "X"),
        }
        for row in rows
    ]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            "group": row["sentence"],
            "sentence": row["sentence"],
            "label": row["label"],
            "target_position": row["target_position"],
            "pos_tag": row.get("pos_tag", "VERB"),
        }
        for row in rows
    ]


def prepare_split(nlp, rows: list[dict[str, str]], output: Path) -> dict[str, int]:
    groups: OrderedDict[str, dict[str, object]] = OrderedDict()
    for row in rows:
        index = int(row["target_position"])
        tokens = row["sentence"].split()
        if not 0 <= index < len(tokens):
            raise ValueError(f"Invalid target index: {row}")
        group = groups.setdefault(
            row["group"],
            {"sentence": row["sentence"], "targets": []},
        )
        group["targets"].append(
            {
                "index": index,
                "label": int(row["label"]),
                "pos": row["pos_tag"],
            }
        )

    docs = [Doc(nlp.vocab, words=item["sentence"].split()) for item in groups.values()]
    parsed = nlp.pipe(docs, batch_size=128)
    records = []
    target_count = 0
    for group, doc in zip(groups.values(), parsed):
        targets = group["targets"]
        target_count += len(targets)
        dependencies = []
        predicted_heads = []
        for token in doc:
            head = 0 if token.head.i == token.i else token.head.i + 1
            predicted_heads.append(head)
            dependencies.append([token.dep_ or "dep", head, token.i + 1])
        records.append(
            {
                "sentence": group["sentence"],
                "tokens": [token.text for token in doc],
                "tags": [token.pos_ or "X" for token in doc],
                "predicted_dependencies": [token.dep_ or "dep" for token in doc],
                "predicted_heads": predicted_heads,
                "dependencies": dependencies,
                "aspect_sentiment": [
                    [doc[item["index"]].text, "metaphor" if item["label"] else "non_meta"]
                    for item in targets
                ],
                "from_to": [[item["index"], item["index"]] for item in targets],
                "target_pos": [item["pos"] for item in targets],
                "ori_sentence": group["sentence"],
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    return {"sentences": len(records), "targets": target_count}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=("vua_all", "vua_verb", "vua_verb_standard", "vua20_extension", "mohx", "trofi"),
        required=True,
    )
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Canonical dataset root; folds live below this path")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    args = parser.parse_args()
    nlp = spacy.load(args.spacy_model, exclude=("ner", "lemmatizer"))

    manifest: dict[str, object] = {"dataset": args.dataset, "parser": args.spacy_model, "splits": {}}
    source = args.source_root.resolve()
    destination = args.output_root.resolve()
    if args.dataset not in ("mohx", "trofi"):
        for split, source_name in (("train", "train.csv"), ("val", "val.csv"), ("test", "test.csv")):
            manifest["splits"][split] = prepare_split(
                nlp, read_csv(source / source_name), destination / f"{split}_rospacy.json"
            )
        manifest_path = destination / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(manifest_path)
        return

    destination_root = destination
    for fold in range(10):
        source_fold = source / f"fold_{fold:02d}"
        destination = destination_root / f"fold_{fold:02d}"
        fold_stats = {}
        for split, source_name in (("train", "train.csv"), ("val", "val.csv"), ("test", "test.csv")):
            fold_stats[split] = prepare_split(
                nlp, read_csv(source_fold / source_name), destination / f"{split}_rospacy.json"
            )
        manifest["splits"][f"fold_{fold:02d}"] = fold_stats
    manifest_path = destination_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
