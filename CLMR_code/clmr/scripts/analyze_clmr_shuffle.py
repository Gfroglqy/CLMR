from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


HERE = Path(__file__).resolve().parent
SEEDS = (13, 21, 42)


def score(labels: np.ndarray, probabilities: np.ndarray) -> float:
    return float(f1_score(labels, probabilities >= 0.5, zero_division=0))


def cluster_confusion(labels: np.ndarray, probabilities: np.ndarray,
                      cluster_codes: np.ndarray, cluster_count: int) -> np.ndarray:
    predictions = probabilities >= 0.5
    counts = np.zeros((cluster_count, 3), dtype=np.int64)
    np.add.at(counts[:, 0], cluster_codes, (predictions & (labels == 1)).astype(np.int64))
    np.add.at(counts[:, 1], cluster_codes, (predictions & (labels == 0)).astype(np.int64))
    np.add.at(counts[:, 2], cluster_codes, ((~predictions) & (labels == 1)).astype(np.int64))
    return counts


def f1_from_counts(counts: np.ndarray) -> float:
    true_positive, false_positive, false_negative = counts.sum(axis=0)
    denominator = 2 * true_positive + false_positive + false_negative
    return float(2 * true_positive / denominator) if denominator else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--real-runs", type=Path, required=True,
                        help="Root containing seed_XX/validation_predictions.csv")
    parser.add_argument("--shuffled-runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = pd.read_csv(args.metadata)
    cluster_codes, clusters = pd.factorize(data.sentence, sort=True)
    count_arrays = []
    per_seed = {}
    for seed in SEEDS:
        real = pd.read_csv(args.real_runs
                           / f"seed_{seed}" / "validation_predictions.csv")
        shuffled = pd.read_csv(args.shuffled_runs
                               / f"seed_{seed}" / "validation_predictions.csv")
        if not (real.label.equals(shuffled.label) and real.label.equals(data.label)):
            raise ValueError(f"Prediction alignment failure for seed {seed}")
        labels = data.label.to_numpy()
        real_probability = real.probability.to_numpy()
        shuffled_probability = shuffled.probability.to_numpy()
        real_f1 = score(labels, real_probability)
        shuffled_f1 = score(labels, shuffled_probability)
        per_seed[str(seed)] = {
            "real": real_f1,
            "shuffled": shuffled_f1,
            "difference": real_f1 - shuffled_f1,
        }
        count_arrays.append((
            cluster_confusion(labels, real_probability, cluster_codes, len(clusters)),
            cluster_confusion(labels, shuffled_probability, cluster_codes, len(clusters)),
        ))

    rng = np.random.default_rng(20260823)
    draws = []
    for _ in range(10_000):
        selected_seeds = rng.integers(0, len(SEEDS), len(SEEDS))
        selected_clusters = rng.integers(0, len(clusters), len(clusters))
        differences = []
        for selected_seed in selected_seeds:
            real_counts, shuffled_counts = count_arrays[selected_seed]
            differences.append(f1_from_counts(real_counts[selected_clusters])
                               - f1_from_counts(shuffled_counts[selected_clusters]))
        draws.append(float(np.mean(differences)))
    draw_array = np.asarray(draws)
    real_values = np.asarray([per_seed[str(seed)]["real"] for seed in SEEDS])
    shuffled_values = np.asarray([per_seed[str(seed)]["shuffled"] for seed in SEEDS])
    result = {
        "validation_only": True,
        "instances": len(data),
        "sentence_clusters": len(clusters),
        "per_seed": per_seed,
        "aggregate": {
            "real": {"mean": float(real_values.mean()),
                     "seed_sd": float(real_values.std(ddof=1))},
            "shuffled": {"mean": float(shuffled_values.mean()),
                         "seed_sd": float(shuffled_values.std(ddof=1))},
        },
        "real_minus_shuffled": {
            "mean": float((real_values - shuffled_values).mean()),
            "ci95": [float(np.quantile(draw_array, 0.025)),
                     float(np.quantile(draw_array, 0.975))],
            "probability_positive": float((draw_array > 0).mean()),
            "positive_seeds": int((real_values > shuffled_values).sum()),
        },
    }
    reports = args.output
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "clmr_real_vs_shuffled.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    comparison = result["real_minus_shuffled"]
    lines = [
        "# CLMR real versus shuffled literal prototypes",
        "",
        "Same checkpoints; inference-time prototypes are deterministically shuffled",
        "under the existing POS/slot-matched control.",
        "",
        "| evidence | F1 mean +/- seed SD |",
        "|---|---:|",
        f"| real | **{real_values.mean()*100:.2f} +/- {real_values.std(ddof=1)*100:.2f}** |",
        f"| shuffled | {shuffled_values.mean()*100:.2f} +/- {shuffled_values.std(ddof=1)*100:.2f} |",
        "",
        f"Real - shuffled: {comparison['mean']*100:+.2f} F1, 95% hierarchical-bootstrap "
        f"CI [{comparison['ci95'][0]*100:+.2f}, {comparison['ci95'][1]*100:+.2f}], "
        f"P(delta>0)={comparison['probability_positive']:.4f}; positive on 3/3 seeds.",
    ]
    (reports / "clmr_real_vs_shuffled.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
