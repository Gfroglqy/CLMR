
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from lemminflect import getLemma


EXPECTED = {"train": 15_516, "val": 1_724, "test": 5_873}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def normalize_sentence(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def lemma(text: str) -> str:
    normalized = re.sub(r"[^A-Za-z'-]+", "", text).lower()
    values = getLemma(normalized, upos="VERB")
    return (values[0] if values else normalized).lower()


def lemma_variants(text: str) -> set[str]:
    normalized = re.sub(r"[^A-Za-z'-]+", "", text).lower()
    return {value.lower() for value in getLemma(normalized, upos="VERB")} | {normalized}


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def write_melbert(rows: list[dict[str, str]], path: Path, split: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["index", "label", "sentence", "POS", "FGPOS", "w_index"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow({
                "index": f"standard-{split}-{index:06d}", "label": row["label"],
                "sentence": row["sentence"], "POS": "VERB", "FGPOS": "VERB",
                "w_index": row["target_position"],
            })


def build_knowledge_lookup(paths: list[Path]):
    lookup: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for path in paths:
        if not path.exists():
            continue
        for row in read_tsv(path):
            target = row.get("target", "")
            for target_lemma in lemma_variants(target):
                lookup[(normalize_sentence(row["sentence"]), target_lemma)].append(row)
    return lookup


def write_contrast(
    rows: list[dict[str, str]], path: Path, split: str,
    released_rows: list[dict[str, str]] | None, lookup,
) -> dict[str, int]:
    fields = ["index", "label", "sentence", "POS", "w_index", "target", "word_sense", "definition"]
    counts = {
        "instances": len(rows), "row_aligned": 0, "lookup": 0, "fallback": 0,
        "target_lemma_mismatches": 0,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for index, row in enumerate(rows):
            target = row["target_word"]
            knowledge = None
            uid = f"standard-{split}-{index:06d}"
            if released_rows is not None:
                released = released_rows[index]
                if int(released["label"]) != int(row["label"]):
                    raise ValueError(f"Released VUA-Verb label mismatch at {split}:{index}")
                if normalize_sentence(released["sentence"]) != normalize_sentence(row["sentence"]):
                    raise ValueError(f"Released VUA-Verb sentence mismatch at {split}:{index}")
                if not (lemma_variants(released.get("target", "")) & lemma_variants(target)):



                    counts["target_lemma_mismatches"] += 1
                knowledge = released
                uid = released["index"]
                counts["row_aligned"] += 1
            else:
                candidates = []
                for target_lemma in lemma_variants(target):
                    candidates.extend(lookup.get((normalize_sentence(row["sentence"]), target_lemma), []))
                label_matched = [candidate for candidate in candidates if int(candidate["label"]) == int(row["label"])]
                if label_matched:
                    knowledge = label_matched[0]
                    uid = knowledge["index"]
                    counts["lookup"] += 1
            if knowledge is None:
                word_sense = target; definition = target; counts["fallback"] += 1
            else:
                word_sense = knowledge.get("word_sense", "").strip() or target
                definition = knowledge.get("definition", "").strip() or target
            writer.writerow({
                "index": uid, "label": row["label"], "sentence": row["sentence"],
                "POS": "VERB", "w_index": row["target_position"], "target": target,
                "word_sense": word_sense, "definition": definition,
            })
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--official-wsd-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    cli = parser.parse_args()

    canonical = cli.output_root / "canonical"
    melbert = cli.output_root / "melbert"
    contrast = cli.output_root / "contrastwsd"
    splits = {name: read_csv(cli.source / f"{name}.csv") for name in EXPECTED}
    for name, expected in EXPECTED.items():
        if len(splits[name]) != expected:
            raise ValueError(f"Unexpected standard VUA-Verb {name} count: {len(splits[name])} != {expected}")
        write_csv(splits[name], canonical / f"{name}.csv")
        write_melbert(splits[name], melbert / f"{'dev' if name == 'val' else name}.tsv", name)

    official = cli.official_wsd_root
    released_train = read_tsv(official / "VUAverb" / "train.tsv")
    released_test = read_tsv(official / "VUAverb" / "test.tsv")
    if len(released_train) != EXPECTED["train"] or len(released_test) != EXPECTED["test"]:
        raise ValueError("Released ContrastWSD VUAverb counts do not match standard split")
    lookup = build_knowledge_lookup([
        official / "VUAverb" / "train.tsv", official / "VUAverb" / "test.tsv",
        official / "VUA18" / "train.tsv", official / "VUA18" / "test.tsv",
        official / "VUA20" / "train.tsv", official / "VUA20" / "test.tsv",
    ])
    contrast_stats = {
        "train": write_contrast(splits["train"], contrast / "train.tsv", "train", released_train, lookup),
        "dev": write_contrast(splits["val"], contrast / "dev.tsv", "val", None, lookup),
        "test": write_contrast(splits["test"], contrast / "test.tsv", "test", released_test, lookup),
    }
    manifest = {
        "dataset": "VUA-Verb standard", "counts": EXPECTED,
        "canonical_source": str(cli.source.resolve()),
        "contrastwsd_knowledge_alignment": contrast_stats,
        "warning": "Not the 20,917/7,152/9,872 VUA-All POS-filtered subset",
    }
    (cli.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
