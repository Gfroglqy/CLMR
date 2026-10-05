from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def confusion(labels: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    prediction = probabilities >= 0.5
    positive = labels == 1
    return np.asarray([
        np.sum(prediction & positive),
        np.sum(prediction & ~positive),
        np.sum(~prediction & positive),
    ], dtype=np.int64)


def f1(counts: np.ndarray) -> float:
    true_positive, false_positive, false_negative = counts.sum(axis=0)
    denominator = 2 * true_positive + false_positive + false_negative
    return float(2 * true_positive / denominator) if denominator else 0.0


def dataset_bootstrap(frame: pd.DataFrame, draws: int, seed: int) -> dict:
    seeds = sorted(frame.seed.unique().tolist())
    per_seed: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    observed = []
    for current_seed in seeds:
        rows = frame[frame.seed.eq(current_seed)].copy()
        codes, clusters = pd.factorize(rows.cluster.astype(str), sort=True)
        first = np.zeros((len(clusters), 3), dtype=np.int64)
        second = np.zeros((len(clusters), 3), dtype=np.int64)
        for cluster_index in range(len(clusters)):
            mask = codes == cluster_index
            first[cluster_index] = confusion(
                rows.label.to_numpy()[mask], rows.first_probability.to_numpy()[mask])
            second[cluster_index] = confusion(
                rows.label.to_numpy()[mask], rows.second_probability.to_numpy()[mask])
        per_seed[int(current_seed)] = (first, second)
        observed.append(f1(first) - f1(second))

    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        chosen_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        differences = []
        for chosen_seed in chosen_seeds:
            first, second = per_seed[int(chosen_seed)]
            chosen_clusters = rng.integers(0, len(first), size=len(first))
            differences.append(f1(first[chosen_clusters]) - f1(second[chosen_clusters]))
        samples[index] = np.mean(differences)
    return {
        "seeds": seeds,
        "observed_mean_f1_difference": float(np.mean(observed)),
        "ci95": [float(np.quantile(samples, 0.025)),
                 float(np.quantile(samples, 0.975))],
        "probability_positive": float(np.mean(samples > 0)),
        "draws": draws,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=20260824)
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    required = {"dataset", "seed", "label", "first_probability",
                "second_probability", "cluster"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing input columns: {sorted(missing)}")
    result = {
        str(dataset): dataset_bootstrap(group, args.draws, args.random_seed)
        for dataset, group in frame.groupby("dataset", sort=True)
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
