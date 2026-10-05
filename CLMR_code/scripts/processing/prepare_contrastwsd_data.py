
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def build_lookup(paths: list[Path]):
    exact: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    by_uid: dict[str, list[dict[str, str]]] = defaultdict(list)
    for path in paths:
        if not path.exists():
            continue
        for row in read_tsv(path):
            uid = row["index"].strip()
            target = row.get("target", "").strip()
            if not target:
                sentence_words = row["sentence"].split()
                position = int(row.get("w_index", row.get("v_index", -1)))
                target = sentence_words[position] if 0 <= position < len(sentence_words) else ""
            row = dict(row)
            row["target"] = target
            exact[(uid, normalize(target))].append(row)
            by_uid[uid].append(row)
    return exact, by_uid


def select_knowledge(uid: str, target: str, exact, by_uid):
    candidates = exact.get((uid, normalize(target)), [])




    if candidates:
        return candidates[0], "exact"


    target_key = normalize(target)
    compatible = [
        row for row in by_uid.get(uid, [])
        if target_key and (normalize(row["target"]).startswith(target_key) or target_key.startswith(normalize(row["target"])))
    ]
    if compatible:
        return compatible[0], "compatible"
    return None, "fallback"


def convert_split(source: Path, destination: Path, exact, by_uid) -> dict[str, int]:
    rows = read_tsv(source)
    counts = {"instances": 0, "exact": 0, "compatible": 0, "fallback": 0}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["index", "label", "sentence", "POS", "w_index", "target", "word_sense", "definition"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            words = row["sentence"].split()
            position = int(row["w_index"])
            if not 0 <= position < len(words):
                raise ValueError(f"Target index out of range in {source}: {row['index']} position={position}")
            target = words[position]
            knowledge, match_type = select_knowledge(row["index"].strip(), target, exact, by_uid)
            if knowledge is None:
                word_sense = target
                definition = target
            else:
                word_sense = str(knowledge.get("word_sense") or "").strip() or target
                definition = str(knowledge.get("definition") or "").strip() or target
            writer.writerow({
                "index": row["index"], "label": row["label"], "sentence": row["sentence"],
                "POS": row["POS"], "w_index": row["w_index"], "target": target,
                "word_sense": word_sense, "definition": definition,
            })
            counts["instances"] += 1
            counts[match_type] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--vua-all-root", type=Path, required=True)
    parser.add_argument("--vua-verb-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    cli = parser.parse_args()

    official = cli.official_root.resolve() / "data"
    manifests = {}
    configurations = {
        "vua_all": (
            cli.vua_all_root.resolve(),
            [
                official / "VUA18" / "train.tsv", official / "VUA18" / "test.tsv",
                official / "VUA20" / "train.tsv", official / "VUA20" / "test.tsv",
            ],
        ),
        "vua_verb": (
            cli.vua_verb_root.resolve(),
            [
                official / "VUAverb" / "train.tsv", official / "VUAverb" / "test.tsv",
                official / "VUA18" / "train.tsv", official / "VUA18" / "test.tsv",
                official / "VUA20" / "train.tsv", official / "VUA20" / "test.tsv",
            ],
        ),
    }
    for dataset, (source_root, knowledge_paths) in configurations.items():
        exact, by_uid = build_lookup(knowledge_paths)
        manifests[dataset] = {
            split: convert_split(
                source_root / f"{source_name}.tsv",
                cli.output_root.resolve() / dataset / f"{split}.tsv",
                exact,
                by_uid,
            )
            for split, source_name in (("train", "train"), ("dev", "dev"), ("test", "test"))
        }

    payload = {
        "source": "authors' Kaggle WSD augmentation aligned to strict VUA rows",
        "fallback": "target token used for both contextual and basic definition when no unique released match exists",
        "datasets": manifests,
        "unsupported": ["mohx", "trofi"],
        "unsupported_reason": "the authors' public augmentation contains no MOH-X or TroFi WSD definitions",
    }
    cli.output_root.mkdir(parents=True, exist_ok=True)
    (cli.output_root / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
