from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from ark_dcr.metrics import binary_metrics

SEEDS = (13, 21, 42)
STAGES = (
    ("ce_only", "CE-only dual encoder", "ce_only"),
    ("matching", "+ literal matching supervision", "match"),
    ("cross_lemma", "+ cross-lemma hard negatives", "match_hn"),
    ("full", "+ R-Drop (full CLMR)", "full"),
)


def cluster_counts(labels: np.ndarray, probability: np.ndarray,
                   codes: np.ndarray, count: int) -> np.ndarray:
    prediction = probability >= .5
    result = np.zeros((count, 3), dtype=np.int64)
    positive = labels.astype(bool)
    np.add.at(result[:, 0], codes, prediction & positive)
    np.add.at(result[:, 1], codes, prediction & ~positive)
    np.add.at(result[:, 2], codes, ~prediction & positive)
    return result


def counts_f1(counts: np.ndarray) -> float:
    tp, fp, fn = counts.sum(axis=0)
    denominator = 2*tp+fp+fn
    return float(2*tp/denominator) if denominator else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True,
                        help="Canonical validation CSV with sentence and label")
    parser.add_argument("--runs", type=Path, required=True,
                        help="Root containing stage/seed_XX/validation_predictions.csv")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = pd.read_csv(args.metadata, usecols=["sentence", "label"])
    labels = metadata.label.to_numpy()
    codes, clusters = pd.factorize(metadata.sentence, sort=True)
    result: dict = {"validation_only": True, "instances": int(len(metadata)),
                    "sentence_clusters": int(len(clusters)), "seeds": list(SEEDS),
                    "stages": {}}
    stored: dict[str, list[np.ndarray]] = {}
    for stage_id, label, directory in STAGES:
        per_seed = {}
        stored[stage_id] = []
        for seed in SEEDS:
            frame = pd.read_csv(args.runs / directory / f"seed_{seed}"
                                / "validation_predictions.csv").sort_values("row_id")
            if not np.array_equal(frame.label.to_numpy(), labels):
                raise ValueError(f"Prediction alignment failure: {stage_id}, seed {seed}")
            metrics = binary_metrics(frame.label, frame.probability)
            literal = float(frame.loc[frame.label.eq(0), "compatibility"].mean())
            metaphor = float(frame.loc[frame.label.eq(1), "compatibility"].mean())
            metrics["literal_matching_gap"] = literal-metaphor
            per_seed[str(seed)] = metrics
            stored[stage_id].append(cluster_counts(
                labels, frame.probability.to_numpy(), codes, len(clusters)))
        aggregate = {}
        for metric in ("f1", "macro_f1", "nll", "brier", "literal_matching_gap"):
            values = np.asarray([per_seed[str(seed)][metric] for seed in SEEDS])
            aggregate[metric] = {"mean": float(values.mean()),
                                 "seed_sd": float(values.std(ddof=1))}
        result["stages"][stage_id] = {"label": label, "parameters": 251_668_996,
                                      "per_seed": per_seed, "aggregate": aggregate}



    rng = np.random.default_rng(20260824)
    draws = np.empty(10_000, dtype=np.float64)
    for iteration in range(len(draws)):
        seed_draw = rng.integers(0, len(SEEDS), len(SEEDS))
        cluster_draw = rng.integers(0, len(clusters), len(clusters))
        draws[iteration] = np.mean([
            counts_f1(stored["full"][index][cluster_draw])-
            counts_f1(stored["ce_only"][index][cluster_draw]) for index in seed_draw])
    paired = []
    for index, seed in enumerate(SEEDS):
        paired.append(counts_f1(stored["full"][index])-counts_f1(stored["ce_only"][index]))
    result["full_minus_ce_only"] = {
        "f1_mean": float(np.mean(paired)),
        "ci95": [float(np.quantile(draws,.025)), float(np.quantile(draws,.975))],
        "probability_positive": float((draws>0).mean()),
        "positive_seeds": int(np.sum(np.asarray(paired)>0)),
    }
    report = args.output
    report.mkdir(parents=True, exist_ok=True)
    (report / "clmr_ablation_3seeds.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# CLMR complete ablation ladder (VUA-All validation)", "",
        "All rows have exactly 251,668,996 parameters and share the same data split, "
        "training epochs, effective batch size, optimizer and three seeds.", "",
        "| stage | F1 mean +/- SD | Macro-F1 mean +/- SD | NLL | Brier | literal-matching gap |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for stage_id, _, _ in STAGES:
        stage = result["stages"][stage_id]
        agg = stage["aggregate"]
        lines.append(
            f"| {stage['label']} | {agg['f1']['mean']*100:.2f} +/- {agg['f1']['seed_sd']*100:.2f} | "
            f"{agg['macro_f1']['mean']*100:.2f} +/- {agg['macro_f1']['seed_sd']*100:.2f} | "
            f"{agg['nll']['mean']:.4f} | {agg['brier']['mean']:.4f} | "
            f"{agg['literal_matching_gap']['mean']:+.3f} |")
    comparison = result["full_minus_ce_only"]
    lines.extend(["",
        f"Full - CE-only: {comparison['f1_mean']*100:+.2f} F1 points; sentence-cluster "
        f"hierarchical-bootstrap 95% CI [{comparison['ci95'][0]*100:+.2f}, "
        f"{comparison['ci95'][1]*100:+.2f}], P(delta>0)={comparison['probability_positive']:.4f}; "
        f"positive on {comparison['positive_seeds']}/3 seeds.", "",
        "The literal-matching gap is a behavioral diagnostic, not a causal explanation.",
    ])
    (report / "clmr_ablation_3seeds.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
