from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import torch
from nltk.corpus import wordnet as wn


def load_glosses(path: Path) -> dict[str, list[str]]:
    entries: dict[str, list[str]] = defaultdict(list)
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) >= 2:
                word, gloss = row[0].strip().casefold(), row[1].strip()
                if word and gloss and gloss not in entries[word]:
                    entries[word].append(gloss)
    return dict(entries)


def candidates(word: str) -> tuple[str, ...]:
    raw = word.casefold().strip(".,!?;:'\"()[]{}")
    values = [raw]
    if raw.endswith("ies") and len(raw) > 3:
        values.append(raw[:-3] + "y")
    for suffix in ("ing", "ed", "es", "s"):
        if raw.endswith(suffix) and len(raw) > len(suffix) + 2:
            values.append(raw[:-len(suffix)])
    return tuple(dict.fromkeys(values))


def verb_gloss(glosses: list[str]) -> str | None:
    return next((text for text in glosses
                 if text.casefold().lstrip(" '\"").startswith("to ")), None)


class ExternalMIPCollator:

    def __init__(self, base, tokenizer, gloss_path: Path, selected_pos: str = "VERB",
                 mode: str = "real", source: str = "csv"):
        self.base = base
        self.tokenizer = tokenizer
        self.glosses = load_glosses(gloss_path)
        self.selected_pos = selected_pos.upper()
        if mode not in {"real", "shuffled"}:
            raise ValueError(f"Unsupported external MIP mode: {mode}")
        self.mode = mode
        if source not in {"csv", "wordnet"}:
            raise ValueError(f"Unsupported external MIP source: {source}")
        self.source = source

    def __call__(self, items):
        batch = self.base(items)
        words, definitions, available = [], [], []
        for item in items:
            example = item.example
            key = None
            definition = None
            if example.pos_tag.upper() == self.selected_pos:
                for candidate in candidates(example.target_word):
                    if self.source == "wordnet":
                        synsets = wn.synsets(candidate, pos=wn.VERB)
                        definition = synsets[0].definition() if synsets else None
                    else:
                        definition = verb_gloss(self.glosses.get(candidate, []))
                    if definition is not None:
                        key = candidate
                        break
            present = definition is not None
            words.append(f"{self.selected_pos.casefold()}: {key or example.target_word.casefold()}")
            definitions.append(definition or "")
            available.append(present)
        if self.mode == "shuffled":
            covered = [index for index,value in enumerate(available) if value]
            if len(covered) > 1:
                original = [definitions[index] for index in covered]
                rotated = original[-1:] + original[:-1]
                for index,definition in zip(covered,rotated):
                    definitions[index] = definition
        encoded = self.tokenizer(
            words + definitions, padding=True, truncation=True, max_length=48,
            return_tensors="pt",
        )
        batch["external_input_ids"] = encoded["input_ids"]
        batch["external_attention_mask"] = encoded["attention_mask"]
        batch["external_available"] = torch.tensor(available, dtype=torch.bool)
        return batch
