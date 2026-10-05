
from __future__ import annotations

import ast
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "artifacts" / "runs"
CLMR = RUNS / "clmr" / "vua18"
DATA = ROOT / "data" / "vua18" / "test.csv"
GENRE = ROOT / "data" / "metadata" / "vua_seq_formatted_test.csv"
DAGS_INPUT = ROOT / "data" / "baseline_inputs" / "dags" / "vua18" / "test_rospacy.json"
OUT = ROOT / "artifacts" / "analysis" / "vua18_subgroups"
SEEDS = (13, 21, 42)
MODELS = {
    "CLMR": None,
    "MelBERT": RUNS / "baselines" / "melbert" / "vua18",
    "DAGS-Auto": RUNS / "baselines" / "dags" / "vua18",
    "RoPPT-Auto": RUNS / "baselines" / "roppt" / "vua18",
    "ContrastWSD-Aligned": RUNS / "baselines" / "contrastwsd" / "vua18",
}

sys.path.insert(0, str(ROOT / "clmr" / "src"))
from ark_dcr.data import canonical_lemma


def expanded_metadata() -> pd.DataFrame:
    rows = pd.read_csv(GENRE)
    values: list[dict[str, object]] = []
    for row in rows.itertuples(index=False):
        labels = ast.literal_eval(row.label_seq)
        pos_tags = ast.literal_eval(row.pos_seq)
        if len(labels) != len(pos_tags):
            raise ValueError(f"Token metadata mismatch: {row.txt_id} {row.sen_ix}")
        for position, (label, pos_tag) in enumerate(zip(labels, pos_tags)):
            values.append({
                "guid": f"test-{row.txt_id} {row.sen_ix} {position}",
                "sentence": row.sentence,
                "target_position": position,
                "label": int(label),
                "pos_tag": str(pos_tag),
                "genre": str(row.genre).strip().title(),
            })
    return pd.DataFrame(values)


def normalize_legacy_text(value: str) -> str:
    result = value
    for _ in range(2):
        if not any(marker in result for marker in ("Ã", "Â")):
            break
        try:
            result = result.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            break
    return result


def load_prediction(model: str, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if model == "CLMR":
        path = CLMR / f"seed_{seed}" / "test_predictions.csv"
        frame = pd.read_csv(path)
        return frame.label.to_numpy(np.int8), frame.probability.to_numpy(float), None
    path = MODELS[model] / f"seed_{seed}" / "test_predictions.npz"
    data = np.load(path, allow_pickle=False)
    if "guid" in data.files:
        guids = data["guid"].astype(str)
    elif model in ("DAGS-Auto", "RoPPT-Auto"):


        guids = None
    elif model == "ContrastWSD-Aligned":
        guids = expanded_metadata().guid.to_numpy()
    else:
        raise ValueError(f"No row identifiers available for {model}")
    return data["label"].astype(np.int8), data["probability"].astype(float), guids


def dags_order_metadata(expanded: pd.DataFrame, batch_size: int = 32) -> pd.DataFrame:
    source = DAGS_INPUT
    records = json.loads(source.read_text(encoding="utf-8"))
    genre_by_sentence = (expanded[["normalized_sentence", "genre"]]
                         .drop_duplicates().set_index("normalized_sentence").genre)
    rows: list[dict[str, object]] = []
    lengths: list[int] = []
    for record in records:
        normalized = normalize_legacy_text(record["ori_sentence"])
        genre = genre_by_sentence.loc[normalized]
        for index, aspect in enumerate(record["aspect_sentiment"]):
            rows.append({
                "sentence": record["ori_sentence"],
                "target_word": aspect[0],
                "target_position": int(record["from_to"][index][0]),
                "pos_tag": record["target_pos"][index],
                "genre": genre,
                "label": 1 if aspect[1] == "metaphor" else 0,
            })
            lengths.append(len(record["tokens"]))
    order: list[int] = []
    for start in range(0, len(rows), batch_size):
        batch_lengths = torch.tensor(lengths[start:start + batch_size])
        order.extend((start + batch_lengths.sort(descending=True).indices).tolist())
    return pd.DataFrame(rows).iloc[order].reset_index(drop=True)


def metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = probability >= 0.5
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "precision": float(precision_score(labels, prediction, zero_division=0)),
        "recall": float(recall_score(labels, prediction, zero_division=0)),
        "f1": float(f1_score(labels, prediction, zero_division=0)),
    }


def add_training_frequency(frame: pd.DataFrame, counts: Counter) -> pd.DataFrame:
    result = frame.copy()
    result["pos_tag"] = result.pos_tag.fillna("UNKNOWN").astype(str).str.upper()
    result["lemma"] = [
        canonical_lemma(str(word), str(pos))
        for word, pos in zip(result.target_word, result.pos_tag)
    ]
    result["train_lemma_pos_frequency"] = [
        counts[(lemma, pos)] for lemma, pos in zip(result.lemma, result.pos_tag)
    ]
    result["lemma_seen"] = np.where(
        result.train_lemma_pos_frequency.gt(0), "seen", "unseen"
    )
    result["frequency_bucket"] = pd.cut(
        result.train_lemma_pos_frequency,
        bins=[-1, 0, 2, 5, 10, 50, np.inf],
        labels=["unseen", "1-2", "3-5", "6-10", "11-50", ">50"],
        ordered=True,
    )
    return result


def main() -> None:
    global RUNS, CLMR, DATA, GENRE, DAGS_INPUT, OUT, MODELS
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=RUNS)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--genre-metadata", type=Path, default=GENRE)
    parser.add_argument("--dags-input", type=Path, default=DAGS_INPUT)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    RUNS = args.runs.resolve()
    CLMR = RUNS / "clmr" / "vua18"
    DATA = args.data.resolve()
    GENRE = args.genre_metadata.resolve()
    DAGS_INPUT = args.dags_input.resolve()
    OUT = args.output.resolve()
    MODELS = {
        "CLMR": None,
        "MelBERT": RUNS / "baselines" / "melbert" / "vua18",
        "DAGS-Auto": RUNS / "baselines" / "dags" / "vua18",
        "RoPPT-Auto": RUNS / "baselines" / "roppt" / "vua18",
        "ContrastWSD-Aligned": RUNS / "baselines" / "contrastwsd" / "vua18",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    canonical = pd.read_csv(DATA)
    train = pd.read_csv(DATA.parent / "train.csv")
    train["pos_tag"] = train.pos_tag.fillna("UNKNOWN").astype(str).str.upper()
    train["lemma"] = [
        canonical_lemma(str(word), str(pos))
        for word, pos in zip(train.target_word, train.pos_tag)
    ]
    training_counts = Counter(zip(train.lemma, train.pos_tag))
    expanded = expanded_metadata()
    canonical["normalized_sentence"] = canonical.sentence.map(normalize_legacy_text)
    expanded["normalized_sentence"] = expanded.sentence.map(normalize_legacy_text)
    keys = ["normalized_sentence", "target_position", "label"]
    conflicts = expanded.groupby(keys).genre.nunique()
    if (conflicts > 1).any():
        raise ValueError("A canonical target maps to conflicting genres")
    canonical["_occurrence"] = canonical.groupby(keys, sort=False).cumcount()
    expanded["_occurrence"] = expanded.groupby(keys, sort=False).cumcount()
    merge_keys = keys + ["_occurrence"]
    lookup = expanded[merge_keys + ["guid", "genre"]]
    metadata = canonical.merge(lookup, on=merge_keys, how="left", validate="one_to_one")
    if metadata[["guid", "genre"]].isna().any().any():
        raise ValueError("Could not map every CLMR row to VUA document metadata")
    metadata = add_training_frequency(metadata, training_counts)
    labels = metadata.label.to_numpy(np.int8)
    canonical_by_guid = metadata.set_index("guid", verify_integrity=True)
    dags_metadata = add_training_frequency(dags_order_metadata(expanded), training_counts)

    raw: list[dict[str, object]] = []
    probabilities: dict[tuple[str, int], np.ndarray] = {}
    for model in MODELS:
        for seed in SEEDS:
            observed, probability, guids = load_prediction(model, seed)
            if model == "CLMR":
                model_metadata = metadata
            elif model in ("DAGS-Auto", "RoPPT-Auto"):
                model_metadata = dags_metadata
            else:
                model_metadata = canonical_by_guid.loc[guids].reset_index()
            model_labels = model_metadata.label.to_numpy(np.int8)
            if not np.array_equal(observed, model_labels):
                raise ValueError(f"Label alignment failed: {model} seed={seed}")
            if model == "CLMR":
                probabilities[(model, seed)] = probability
            elif guids is not None:
                probabilities[(model, seed)] = pd.Series(probability, index=guids).loc[metadata.guid].to_numpy()
            for kind, column in (
                ("pos", "pos_tag"),
                ("genre", "genre"),
                ("lemma_seen", "lemma_seen"),
                ("train_frequency", "frequency_bucket"),
            ):
                for group, indices in model_metadata.groupby(column, sort=True).groups.items():
                    idx = np.asarray(list(indices), dtype=int)
                    row = {"model": model, "seed": seed, "group_type": kind,
                           "group": group, "instances": len(idx),
                           "positive_rate": float(model_labels[idx].mean())}
                    row.update(metrics(model_labels[idx], probability[idx]))
                    raw.append(row)

    raw_frame = pd.DataFrame(raw)
    raw_frame.to_csv(OUT / "vua18_subgroups_raw.csv", index=False)
    summary = (raw_frame.groupby(["model", "group_type", "group"], sort=True)
               .agg(instances=("instances", "first"), positive_rate=("positive_rate", "first"),
                    accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
                    precision_mean=("precision", "mean"), precision_std=("precision", "std"),
                    recall_mean=("recall", "mean"), recall_std=("recall", "std"),
                    f1_mean=("f1", "mean"), f1_std=("f1", "std"))
               .reset_index())
    summary.to_csv(OUT / "vua18_subgroups_summary.csv", index=False)

    clmr = np.mean([probabilities[("CLMR", seed)] for seed in SEEDS], axis=0)
    melbert = np.mean([probabilities[("MelBERT", seed)] for seed in SEEDS], axis=0)
    clmr_ok = (clmr >= 0.5) == labels
    melbert_ok = (melbert >= 0.5) == labels
    cases = metadata[[
        "sentence", "target_word", "target_position", "pos_tag", "genre", "label",
        "lemma", "train_lemma_pos_frequency", "lemma_seen", "frequency_bucket",
    ]].copy()
    cases["clmr_probability"] = clmr
    cases["melbert_probability"] = melbert
    cases["case_type"] = np.where(clmr_ok & ~melbert_ok, "CLMR_corrected",
                                   np.where(~clmr_ok & melbert_ok, "CLMR_introduced", "unchanged"))
    cases["contrast"] = np.abs(clmr - melbert)
    selected = (cases[cases.case_type != "unchanged"].sort_values(
        ["case_type", "contrast"], ascending=[True, False]).groupby("case_type").head(50))
    selected.to_csv(OUT / "clmr_vs_melbert_case_candidates.csv", index=False)
    review = selected.copy()
    review["human_category"] = ""
    review["annotation_confidence"] = ""
    review["annotation_notes"] = ""
    review.to_csv(OUT / "clmr_vs_melbert_manual_review_100.csv", index=False,
                  encoding="utf-8-sig")

    payload = {
        "instances": int(len(metadata)), "models": list(MODELS), "seeds": list(SEEDS),
        "genres": sorted(metadata.genre.unique().tolist()),
        "pos_groups": sorted(metadata.pos_tag.unique().tolist()),
        "frequency_groups": ["unseen", "1-2", "3-5", "6-10", "11-50", ">50"],
        "case_candidates": int(len(selected)),
    }
    (OUT / "manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
