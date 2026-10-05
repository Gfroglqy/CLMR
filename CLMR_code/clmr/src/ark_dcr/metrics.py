from __future__ import annotations

import numpy as np


def _f1(y: np.ndarray, pred: np.ndarray, positive: int) -> float:
    tp = int(((y == positive) & (pred == positive)).sum())
    fp = int(((y != positive) & (pred == positive)).sum())
    fn = int(((y == positive) & (pred != positive)).sum())
    return 2.0 * tp / max(2 * tp + fp + fn, 1)


def binary_metrics(labels, probabilities) -> dict[str, float]:
    y = np.asarray(labels, dtype=np.int64)
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-7, 1 - 1e-7)
    pred = (p >= 0.5).astype(np.int64)
    positive_f1 = _f1(y, pred, 1)
    return {
        "accuracy": float((y == pred).mean()),
        "precision": float(((y == 1) & (pred == 1)).sum() / max((pred == 1).sum(), 1)),
        "recall": float(((y == 1) & (pred == 1)).sum() / max((y == 1).sum(), 1)),
        "f1": positive_f1,
        "macro_f1": 0.5 * (positive_f1 + _f1(y, pred, 0)),
        "nll": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
        "brier": float(np.square(p - y).mean()),
    }

