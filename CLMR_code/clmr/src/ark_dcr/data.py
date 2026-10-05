from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
from lemminflect import getLemma
from torch.utils.data import Dataset


@dataclass(frozen=True)
class LiteralExample:
    row_id: int
    sentence: str
    label: int
    target_position: int
    target_word: str
    pos_tag: str
    gloss: str
    example_sentence: str


def load_examples(path: str | Path) -> list[LiteralExample]:
    result: list[LiteralExample] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row_id, row in enumerate(csv.DictReader(handle)):
            result.append(
                LiteralExample(
                    row_id=row_id,
                    sentence=row["sentence"],
                    label=int(row["label"]),
                    target_position=int(row["target_position"]),
                    target_word=row["target_word"],
                    pos_tag=row["pos_tag"],
                    gloss=row.get("gloss", "") or "",
                    example_sentence=row.get("eg_sent", "") or "",
                )
            )
    return result


class LiteralEvidenceDataset(Dataset):
    def __init__(self, examples: list[LiteralExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> LiteralExample:
        return self.examples[index]


@dataclass(frozen=True)
class LiteralPrototype:
    text: str
    source_id: int
    source_row_id: int


@dataclass(frozen=True)
class PrototypeExample:
    example: LiteralExample
    prototypes: tuple[LiteralPrototype, ...]


@lru_cache(maxsize=32768)
def canonical_lemma(word: str, pos: str) -> str:

    normalized = re.sub(r"[^A-Za-z'-]+", "", word).lower()
    if not normalized:
        return word.lower()
    upos = pos.upper()
    if upos in {"VERB", "NOUN", "ADJ", "ADV"}:
        lemmas = getLemma(normalized, upos=upos)
        if lemmas:
            return lemmas[0].lower()
    return normalized


def _tokens(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"[A-Za-z]+", text.lower()))


class LiteralPrototypeMemory:

    QUERY_GLOSS = 1
    QUERY_EXAMPLE = 2
    TRAIN_GLOSS = 3
    TRAIN_EXAMPLE = 4
    TRAIN_CONTEXT = 5

    def __init__(
        self,
        train_examples: list[LiteralExample],
        max_candidates: int = 64,
        anchor_first: bool = True,
    ):
        self.anchor_first = anchor_first
        pools: dict[tuple[str, str], dict[str, LiteralPrototype]] = {}
        for example in train_examples:
            if example.label != 0:
                continue
            key = (canonical_lemma(example.target_word, example.pos_tag), example.pos_tag.upper())
            bucket = pools.setdefault(key, {})
            candidates = (
                (example.gloss, self.TRAIN_GLOSS),
                (example.example_sentence, self.TRAIN_EXAMPLE),
                (example.sentence, self.TRAIN_CONTEXT),
            )
            for text, source_id in candidates:
                normalized = " ".join(text.split())
                if normalized:
                    bucket.setdefault(
                        normalized.lower(),
                        LiteralPrototype(normalized, source_id, example.row_id),
                    )

        self.pools: dict[tuple[str, str], tuple[LiteralPrototype, ...]] = {}
        for key, unique in pools.items():
            ordered = sorted(
                unique.values(),
                key=lambda item: hashlib.sha1(item.text.encode("utf-8")).hexdigest(),
            )
            self.pools[key] = tuple(ordered[:max_candidates])

    def retrieve(
        self,
        example: LiteralExample,
        top_k: int,
        exclude_training_row: bool = False,
    ) -> tuple[LiteralPrototype, ...]:
        key = (canonical_lemma(example.target_word, example.pos_tag), example.pos_tag.upper())

        query_candidates: list[LiteralPrototype] = []
        if example.example_sentence.strip():
            query_candidates.append(
                LiteralPrototype(" ".join(example.example_sentence.split()), self.QUERY_EXAMPLE, -1)
            )
        if example.gloss.strip():
            query_candidates.append(
                LiteralPrototype(" ".join(example.gloss.split()), self.QUERY_GLOSS, -1)
            )
        memory_candidates = list(self.pools.get(key, ()))

        query_words = example.sentence.split()
        context = " ".join(
            word for i, word in enumerate(query_words) if i != example.target_position
        )
        query_tokens = _tokens(context)
        deduplicated: dict[str, LiteralPrototype] = {}
        for candidate in memory_candidates:
            if exclude_training_row and candidate.source_row_id == example.row_id:
                continue
            deduplicated.setdefault(candidate.text.lower(), candidate)

        def score(candidate: LiteralPrototype) -> tuple[float, int, str]:
            candidate_tokens = _tokens(candidate.text)
            union = query_tokens | candidate_tokens
            jaccard = len(query_tokens & candidate_tokens) / max(len(union), 1)


            source_bonus = 1 if candidate.source_id in {self.QUERY_GLOSS, self.QUERY_EXAMPLE} else 0
            stable = hashlib.sha1(candidate.text.encode("utf-8")).hexdigest()
            return (jaccard, source_bonus, stable)

        ranked_memory = sorted(deduplicated.values(), key=score, reverse=True)
        if not self.anchor_first:
            all_candidates: dict[str, LiteralPrototype] = {}

            legacy_query = []
            if example.gloss.strip():
                legacy_query.append(
                    LiteralPrototype(" ".join(example.gloss.split()), self.QUERY_GLOSS, -1)
                )
            if example.example_sentence.strip():
                legacy_query.append(
                    LiteralPrototype(
                        " ".join(example.example_sentence.split()), self.QUERY_EXAMPLE, -1
                    )
                )
            for candidate in [*legacy_query, *ranked_memory]:
                all_candidates.setdefault(candidate.text.lower(), candidate)
            return tuple(sorted(all_candidates.values(), key=score, reverse=True)[:top_k])
        ordered: list[LiteralPrototype] = []
        seen: set[str] = set()
        for candidate in [*query_candidates, *ranked_memory]:
            normalized = candidate.text.lower()
            if normalized not in seen:
                ordered.append(candidate)
                seen.add(normalized)
        return tuple(ordered[:top_k])

    def materialize(
        self,
        examples: list[LiteralExample],
        top_k: int,
        training: bool = False,
    ) -> list[PrototypeExample]:
        return [
            PrototypeExample(
                example=example,
                prototypes=self.retrieve(example, top_k, exclude_training_row=training),
            )
            for example in examples
        ]


class PrototypeEvidenceDataset(Dataset):
    def __init__(self, examples: list[PrototypeExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> PrototypeExample:
        return self.examples[index]


class LiteralEvidenceCollator:

    POS_TO_ID = {"NOUN": 1, "VERB": 2, "ADJ": 3, "ADV": 4, "ADP": 5, "DET": 6,
                 "PRON": 7, "PROPN": 8, "AUX": 9, "CCONJ": 10, "SCONJ": 11,
                 "NUM": 12, "PART": 13, "INTJ": 14, "PUNCT": 15, "X": 16}

    def __init__(self, tokenizer, max_context_length: int = 150, max_evidence_length: int = 96):
        if not tokenizer.is_fast:
            raise ValueError("A fast tokenizer is required for word_ids alignment")
        self.tokenizer = tokenizer
        self.max_context_length = max_context_length
        self.max_evidence_length = max_evidence_length

    def __call__(self, examples: list[LiteralExample]) -> dict:
        words = [x.sentence.split() for x in examples]
        context = self.tokenizer(
            words,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=self.max_context_length,
            return_tensors="pt",
        )
        target_masks = torch.zeros_like(context["attention_mask"], dtype=torch.bool)
        for i, example in enumerate(examples):
            if not 0 <= example.target_position < len(words[i]):
                raise ValueError(f"Invalid target position in row {example.row_id}")
            word_ids = context.word_ids(batch_index=i)
            target_masks[i] = torch.tensor(
                [wid == example.target_position for wid in word_ids], dtype=torch.bool
            )
            if not target_masks[i].any():
                raise ValueError(f"Target truncated in row {example.row_id}")

        literal_text = [x.example_sentence if x.example_sentence.strip() else x.gloss for x in examples]
        evidence = self.tokenizer(
            [x.target_word for x in examples],
            literal_text,
            padding=True,
            truncation="only_second",
            max_length=self.max_evidence_length,
            return_tensors="pt",
        )
        literal_target_masks = torch.zeros_like(evidence["attention_mask"], dtype=torch.bool)
        example_masks = torch.zeros_like(evidence["attention_mask"], dtype=torch.bool)
        for i in range(len(examples)):
            sequence_ids = evidence.sequence_ids(i)
            literal_target_masks[i] = torch.tensor([sid == 0 for sid in sequence_ids], dtype=torch.bool)
            example_masks[i] = torch.tensor([sid == 1 for sid in sequence_ids], dtype=torch.bool)

        return {
            "context_input_ids": context["input_ids"],
            "context_attention_mask": context["attention_mask"],
            "target_mask": target_masks,
            "evidence_input_ids": evidence["input_ids"],
            "evidence_attention_mask": evidence["attention_mask"],
            "literal_target_mask": literal_target_masks,
            "example_mask": example_masks,
            "pos_ids": torch.tensor([self.POS_TO_ID.get(x.pos_tag.upper(), 0) for x in examples]),
            "lemma_ids": torch.tensor([int(hashlib.sha1(
                canonical_lemma(x.target_word,x.pos_tag).encode("utf-8")).hexdigest()[:15],16)
                for x in examples],dtype=torch.long),
            "labels": torch.tensor([x.label for x in examples]),
            "row_ids": torch.tensor([x.row_id for x in examples]),
        }


class PrototypeEvidenceCollator(LiteralEvidenceCollator):

    def __init__(
        self,
        tokenizer,
        prototype_count: int,
        max_context_length: int = 150,
        max_evidence_length: int = 96,
    ):
        super().__init__(tokenizer, max_context_length, max_evidence_length)
        if prototype_count < 2:
            raise ValueError("prototype_count must be at least 2")
        self.prototype_count = prototype_count

    def __call__(self, items: list[PrototypeExample]) -> dict:
        examples = [item.example for item in items]
        words = [x.sentence.split() for x in examples]
        context = self.tokenizer(
            words,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=self.max_context_length,
            return_tensors="pt",
        )
        target_masks = torch.zeros_like(context["attention_mask"], dtype=torch.bool)
        for i, example in enumerate(examples):
            if not 0 <= example.target_position < len(words[i]):
                raise ValueError(f"Invalid target position in row {example.row_id}")
            word_ids = context.word_ids(batch_index=i)
            target_masks[i] = torch.tensor(
                [wid == example.target_position for wid in word_ids], dtype=torch.bool
            )
            if not target_masks[i].any():
                raise ValueError(f"Target truncated in row {example.row_id}")

        flat_targets: list[str] = []
        flat_texts: list[str] = []
        valid = torch.zeros((len(items), self.prototype_count), dtype=torch.bool)
        source_ids = torch.zeros((len(items), self.prototype_count), dtype=torch.long)
        for i, item in enumerate(items):
            for j in range(self.prototype_count):
                flat_targets.append(item.example.target_word)
                if j < len(item.prototypes):
                    prototype = item.prototypes[j]
                    flat_texts.append(prototype.text)
                    valid[i, j] = True
                    source_ids[i, j] = prototype.source_id
                else:
                    flat_texts.append("")

        evidence = self.tokenizer(
            flat_targets,
            flat_texts,
            padding=True,
            truncation="only_second",
            max_length=self.max_evidence_length,
            return_tensors="pt",
        )
        literal_masks = torch.zeros_like(evidence["attention_mask"], dtype=torch.bool)
        example_masks = torch.zeros_like(evidence["attention_mask"], dtype=torch.bool)
        for i in range(len(flat_targets)):
            sequence_ids = evidence.sequence_ids(i)
            literal_masks[i] = torch.tensor([sid == 0 for sid in sequence_ids], dtype=torch.bool)
            example_masks[i] = torch.tensor([sid == 1 for sid in sequence_ids], dtype=torch.bool)

        return {
            "context_input_ids": context["input_ids"],
            "context_attention_mask": context["attention_mask"],
            "target_mask": target_masks,
            "evidence_input_ids": evidence["input_ids"],
            "evidence_attention_mask": evidence["attention_mask"],
            "literal_target_mask": literal_masks,
            "example_mask": example_masks,
            "prototype_valid_mask": valid,
            "prototype_source_ids": source_ids,
            "pos_ids": torch.tensor([self.POS_TO_ID.get(x.pos_tag.upper(), 0) for x in examples]),
            "lemma_ids": torch.tensor([int(hashlib.sha1(
                canonical_lemma(x.target_word,x.pos_tag).encode("utf-8")).hexdigest()[:15],16)
                for x in examples],dtype=torch.long),
            "labels": torch.tensor([x.label for x in examples]),
            "row_ids": torch.tensor([x.row_id for x in examples]),
        }
