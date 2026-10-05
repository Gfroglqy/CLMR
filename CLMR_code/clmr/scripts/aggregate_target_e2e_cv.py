from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


HERE = Path(__file__).resolve().parent
OUTPUT_ROOT = HERE / "outputs" / "clmr_target_e2e_cv"
REPORT_ROOT = HERE / "reports"
DATASETS = ("mohx", "trofi")
SEEDS = (13, 21, 42)


def metrics_from_frame(frame: pd.DataFrame) -> dict[str, float]:
    labels = frame["label"].astype(int).to_numpy()
    predictions = (frame["probability"].astype(float).to_numpy() >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }


def aggregate_dataset(dataset: str) -> dict:
    seed_rows: list[dict] = []
    fold_rows: list[dict] = []
    for seed in SEEDS:
        frames: list[pd.DataFrame] = []
        for fold in range(10):
            run_dir = OUTPUT_ROOT / dataset / f"fold_{fold:02d}" / f"seed_{seed}"
            prediction_path = run_dir / "test_predictions.csv"
            metrics_path = run_dir / "metrics.json"
            if not prediction_path.exists() or not metrics_path.exists():
                raise FileNotFoundError(f"Incomplete run: {run_dir}")
            frame = pd.read_csv(prediction_path)
            frame.insert(0, "fold", fold)
            frame.insert(1, "seed", seed)
            frames.append(frame)
            summary = json.loads(metrics_path.read_text(encoding="utf-8"))
            fold_rows.append({
                "dataset": dataset,
                "seed": seed,
                "fold": fold,
                "instances": len(frame),
                **{key: float(summary["test"][key]) for key in
                   ("accuracy", "precision", "recall", "f1", "macro_f1", "nll", "brier")},
            })
        pooled = pd.concat(frames, ignore_index=True)
        seed_rows.append({
            "dataset": dataset,
            "seed": seed,
            "instances": len(pooled),
            **metrics_from_frame(pooled),
        })

    seed_frame = pd.DataFrame(seed_rows)
    fold_frame = pd.DataFrame(fold_rows)
    aggregate = {}
    for metric in ("accuracy", "precision", "recall", "f1", "macro_f1"):
        values = seed_frame[metric].to_numpy(float)
        aggregate[metric] = {
            "mean": float(values.mean()),
            "sample_sd": float(values.std(ddof=1)),
        }
    return {
        "protocol": (
            "Target-only end-to-end 10-fold cross-validation. Each fold selects "
            "its checkpoint on fold-local validation data and evaluates test once. "
            "Primary metrics pool all ten held-out folds within each seed, then "
            "report mean and sample SD over seeds."
        ),
        "seeds": seed_rows,
        "aggregate": aggregate,
        "folds": fold_rows,
    }


def render_markdown(results: dict[str, dict]) -> str:
    lines = [
        "# CLMR target-only end-to-end cross-validation",
        "",
        "Primary unit of replication: random seed. The ten test folds are pooled",
        "within each seed and are not treated as ten independent model replicates.",
        "",
        "| dataset | Accuracy mean +/- SD | Precision mean +/- SD | Recall mean +/- SD | F1 mean +/- SD | Macro-F1 mean +/- SD | seeds |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, result in results.items():
        agg = result["aggregate"]
        cell = lambda key: f'{100*agg[key]["mean"]:.2f} +/- {100*agg[key]["sample_sd"]:.2f}'
        lines.append(
            f"| {dataset.upper()} | {cell('accuracy')} | {cell('precision')} | "
            f"{cell('recall')} | **{cell('f1')}** | {cell('macro_f1')} | 3 |"
        )
    lines.extend(["", "## Per-seed pooled out-of-fold metrics", ""])
    for dataset, result in results.items():
        lines.extend([
            f"### {dataset.upper()}", "",
            "| seed | instances | Accuracy | Precision | Recall | F1 | Macro-F1 |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in result["seeds"]:
            lines.append(
                f"| {row['seed']} | {row['instances']} | {100*row['accuracy']:.2f} | "
                f"{100*row['precision']:.2f} | {100*row['recall']:.2f} | "
                f"{100*row['f1']:.2f} | {100*row['macro_f1']:.2f} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    global OUTPUT_ROOT, REPORT_ROOT, DATASETS, SEEDS
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True,
                        help="Root containing <dataset>/fold_XX/seed_YY runs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["mohx", "trofi"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[13, 21, 42])
    args = parser.parse_args()
    OUTPUT_ROOT = args.runs.resolve()
    REPORT_ROOT = args.output.resolve()
    DATASETS = tuple(args.datasets)
    SEEDS = tuple(args.seeds)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    results = {dataset: aggregate_dataset(dataset) for dataset in DATASETS}
    (REPORT_ROOT / "clmr_target_e2e_cv_3seeds.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    (REPORT_ROOT / "clmr_target_e2e_cv_3seeds.md").write_text(
        render_markdown(results), encoding="utf-8"
    )
    pd.DataFrame(
        row for result in results.values() for row in result["folds"]
    ).to_csv(REPORT_ROOT / "clmr_target_e2e_cv_fold_metrics.csv", index=False)
    print(REPORT_ROOT / "clmr_target_e2e_cv_3seeds.md")


if __name__ == "__main__":
    main()
