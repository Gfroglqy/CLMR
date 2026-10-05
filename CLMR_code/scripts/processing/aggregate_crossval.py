
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)


def expected_calibration_error(labels: np.ndarray, probability: np.ndarray, bins: int = 15) -> float:
    predictions = (probability >= 0.5).astype(np.int64)
    confidence = np.where(predictions == 1, probability, 1.0 - probability)
    correctness = (predictions == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        selected = ((confidence >= lower) if index == 0 else (confidence > lower)) & (confidence <= upper)
        if np.any(selected):
            value += np.mean(selected) * abs(np.mean(correctness[selected]) - np.mean(confidence[selected]))
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = [args.run_root / f"fold_{fold:02d}" / "test_predictions.npz" for fold in range(10)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing outer-fold predictions: {missing}")
    arrays = [np.load(path) for path in paths]
    labels = np.concatenate([item["label"] for item in arrays]).astype(np.int64)
    probabilities = np.concatenate([item["probability"] for item in arrays]).astype(np.float64)
    predictions = (probabilities >= 0.5).astype(np.int64)
    metrics = {
        "outer_folds": 10,
        "instances": int(labels.size),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "nll": float(log_loss(labels, np.column_stack((1.0 - probabilities, probabilities)), labels=[0, 1])),
        "brier": float(brier_score_loss(labels, probabilities)),
        "ece": expected_calibration_error(labels, probabilities),
    }
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
