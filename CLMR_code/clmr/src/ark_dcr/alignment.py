
from __future__ import annotations

import ast
import collections
import csv
import re
from pathlib import Path


def normalize(value: str) -> str:

    return re.sub(r"[^\x20-\x7e]+", "<U>", value)


def read_csv(path: str | Path) -> list[dict[str, str]]:

    raw = Path(path).read_bytes()
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return list(csv.DictReader(raw.decode(encoding).splitlines()))
        except UnicodeDecodeError:
            continue
    raise UnicodeError(path)


def align(
    dcr_prediction: str | Path,
    dcr_raw: str | Path,
    lee_prediction: str | Path,
    target_raw: str | Path,
) -> list[dict]:

    dcr = read_csv(dcr_prediction)
    source = read_csv(dcr_raw)
    lee_prob = {
        int(row["row_id"]): float(row["probability"])
        for row in read_csv(lee_prediction)
    }
    target_rows = read_csv(target_raw)
    lookup: dict[tuple, list[tuple[int, dict]]] = collections.defaultdict(list)
    for row_id, row in enumerate(target_rows):
        key = (
            normalize(row["sentence"]),
            int(row["target_position"]),
            normalize(row["target_word"]),
            int(row["label"]),
        )
        lookup[key].append((row_id, row))

    aligned: list[dict] = []
    index = 0
    for sentence in source:
        words = sentence["sentence"].split()
        labels = list(map(int, ast.literal_eval(sentence["label_seq"])))
        poses = ast.literal_eval(sentence["pos_seq"])
        for word_index, (word, label, pos) in enumerate(zip(words, labels, poses)):
            dcr_row = dcr[index]
            index += 1
            key = (normalize(sentence["sentence"]), word_index, normalize(word), label)
            if not lookup[key]:
                raise RuntimeError(f"Cannot align {key}")
            row_id, target_row = lookup[key].pop(0)
            aligned.append(
                {
                    "label": label,
                    "dcr": float(dcr_row["probability"]),
                    "wsd_transition_prediction": (
                        float(dcr_row["wsd_transition_prediction"])
                        if dcr_row.get("wsd_transition_prediction", "").strip()
                        else None
                    ),
                    "lee": lee_prob[row_id],
                    "pos": pos,
                    "cluster_id": dcr_row.get("cluster_id", str(index)),
                    "row_id": row_id,
                    "knowledge_available": bool(
                        target_row.get("eg_sent", "").strip()
                        or target_row.get("gloss", "").strip()
                    ),
                }
            )

    if index != len(dcr) or len(aligned) != len(target_rows):
        raise RuntimeError("Cardinality mismatch after alignment")
    return aligned
