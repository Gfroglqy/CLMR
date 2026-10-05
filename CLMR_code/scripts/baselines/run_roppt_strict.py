"""Strict evaluator for the released RoPPT architecture.

The released trainer evaluates the test set whenever validation improves.  This
adapter keeps RoPPT's model computation and the shared dependency-graph input
pipeline, but selects one checkpoint on validation F1 and evaluates test once.
Only compact metrics and predictions are written to disk.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "processing"))

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup


def expected_calibration_error(labels: np.ndarray, probability: np.ndarray, bins: int = 15) -> float:
    prediction = (probability >= 0.5).astype(np.int64)
    confidence = np.where(prediction == 1, probability, 1.0 - probability)
    correct = (prediction == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        lower = confidence >= edges[index] if index == 0 else confidence > edges[index]
        selected = lower & (confidence <= edges[index + 1])
        if np.any(selected):
            result += np.mean(selected) * abs(np.mean(correct[selected]) - np.mean(confidence[selected]))
    return float(result)


def unpack(batch, device: torch.device) -> dict[str, torch.Tensor]:
    batch = tuple(item.to(device) for item in batch)
    return {
        "input_ids": batch[4],
        "attention_mask": batch[15],
        "token_type_ids": batch[5],
        "labels": batch[10],
        "input_ids_2": batch[2],
        "attention_mask_2": batch[16],
        "target_mask_2": batch[18],
        "word_indexer": batch[1],
        "dep_tags": batch[7],
        "target_mask": batch[0],
        "neighbor_mask": batch[17],
    }


def forward(model: torch.nn.Module, values: dict[str, torch.Tensor]) -> torch.Tensor:
    return model(
        values["input_ids"],
        values["input_ids_2"],
        target_mask=values["target_mask"],
        target_mask_2=values["target_mask_2"],
        attention_mask_2=values["attention_mask_2"],
        token_type_ids=values["token_type_ids"],
        attention_mask=values["attention_mask"],
        word_indexer=values["word_indexer"],
        dep_tags=values["dep_tags"],
        neighbor_mask=values["neighbor_mask"],
    )


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_labels, all_probabilities = [], []
    with torch.no_grad():
        for batch in loader:
            values = unpack(batch, device)
            log_probability = forward(model, values)
            all_labels.append(values["labels"].cpu().numpy())
            all_probabilities.append(torch.exp(log_probability)[:, 1].cpu().numpy())
    labels = np.concatenate(all_labels).astype(np.int64)
    probability = np.concatenate(all_probabilities).astype(np.float64)
    prediction = (probability >= 0.5).astype(np.int64)
    metrics = {
        "accuracy": float(accuracy_score(labels, prediction)),
        "precision": float(precision_score(labels, prediction, zero_division=0)),
        "recall": float(recall_score(labels, prediction, zero_division=0)),
        "f1": float(f1_score(labels, prediction, zero_division=0)),
        "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
        "nll": float(log_loss(labels, np.column_stack((1.0 - probability, probability)), labels=[0, 1])),
        "brier": float(brier_score_loss(labels, probability)),
        "ece": expected_calibration_error(labels, probability),
    }
    return metrics, labels, probability, prediction


def load_dependency_module(dags_repo: Path):
    """Reuse the already validated automatic-dependency compatibility layer."""
    module_path = dags_repo.resolve() / "datasets.py"
    spec = importlib.util.spec_from_file_location("strict_dependency_datasets", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load dependency dataset module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dependency-repo", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--class-weight", type=float, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--forward-check", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--encoder-init", type=Path)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--save-encoder", type=Path)
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--benchmark-output", type=Path)
    parser.add_argument("--benchmark-batches", type=int, default=50)
    parser.add_argument("--warmup-batches", type=int, default=10)
    parser.add_argument("--benchmark-repeats", type=int, default=3)
    cli = parser.parse_args()

    sys.path.insert(0, str(cli.repo.resolve()))
    from utils import Config
    from trainer import get_collate_fn
    from main import load_pretrained_model

    dependency_datasets = load_dependency_module(cli.dependency_repo)
    args = Config(str(cli.repo.resolve()))
    args.data_dir = str(cli.data_dir.resolve())
    args.output_dir = args.data_dir
    args.outputdir = args.data_dir
    args.error_save_dir = args.data_dir
    args.dataset_name = cli.dataset
    args.bert_model = str(cli.backbone.resolve())
    args.model_type = "MELBERT_GAT"
    args.embedding_type = "melbert"
    args.seed = cli.seed
    args.num_train_epoch = cli.epochs
    args.per_gpu_train_batch_size = cli.batch_size
    args.per_gpu_eval_batch_size = max(cli.batch_size, 32)
    args.class_weight = cli.class_weight
    args.num_labels = 2
    selected_device = ("cuda" if torch.cuda.is_available() else "cpu") if cli.device == "auto" else cli.device
    args.device = torch.device(selected_device)
    args.n_gpu = 1 if selected_device == "cuda" else 0
    args.tokenizer = AutoTokenizer.from_pretrained(args.bert_model, do_lower_case=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_set, test_set, dev_set, _, dep_vocab, pos_vocab = dependency_datasets.load_datasets_and_vocabs(args)
    args.dep_tag_num = max(int(args.dep_tag_num), int(dep_vocab["len"]))
    args.pos_tag_num = max(int(args.pos_tag_num), int(pos_vocab["len"]))
    collate = get_collate_fn(args)
    train_loader = DataLoader(train_set, sampler=RandomSampler(train_set), batch_size=args.per_gpu_train_batch_size, collate_fn=collate)
    dev_loader = DataLoader(dev_set, sampler=SequentialSampler(dev_set), batch_size=args.per_gpu_eval_batch_size, collate_fn=collate)
    test_loader = DataLoader(test_set, sampler=SequentialSampler(test_set), batch_size=args.per_gpu_eval_batch_size, collate_fn=collate)

    if cli.check_only:
        batch = next(iter(train_loader))
        payload = {
            "train": len(train_set), "validation": len(dev_set), "test": len(test_set),
            "dep_vocab": int(dep_vocab["len"]), "pos_vocab": int(pos_vocab["len"]),
            "batch_fields": len(batch), "batch_size": int(batch[0].shape[0]),
        }
        if cli.forward_check:
            model = load_pretrained_model(args)
            model.eval()
            with torch.no_grad():
                logits = forward(model, unpack(batch, args.device))
            payload["logits_shape"] = list(logits.shape)
            payload["finite"] = bool(torch.isfinite(logits).all().item())
            payload["device"] = str(args.device)
        print(json.dumps(payload, indent=2))
        return

    model = load_pretrained_model(args)
    if cli.benchmark_only:
        from efficiency_utils import benchmark_cuda, collect_batches, model_size

        batches = collect_batches(test_loader, cli.benchmark_batches, args.device)
        result = {
            "model": "RoPPT-Auto-spacy-sm",
            "hardware": torch.cuda.get_device_name(0),
            "precision": "bf16",
            "batch_size": int(args.per_gpu_eval_batch_size),
            "max_context_length": int(args.max_seq_length),
            **model_size(model),
            **benchmark_cuda(
                model, batches, lambda current, batch: forward(current, unpack(batch, args.device)),
                warmup_batches=cli.warmup_batches, repeats=cli.benchmark_repeats,
            ),
        }
        if cli.benchmark_output is None:
            raise ValueError("--benchmark-output is required with --benchmark-only")
        cli.benchmark_output.parent.mkdir(parents=True, exist_ok=True)
        cli.benchmark_output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return
    if cli.encoder_init:
        encoder_state = torch.load(cli.encoder_init, map_location="cpu", weights_only=True)
        model.encoder.load_state_dict(encoder_state, strict=True)
    if cli.freeze_encoder:
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
    grouped = [
        {"params": [p for name, p in model.named_parameters() if p.requires_grad and not any(key in name for key in ("bias", "LayerNorm.bias", "LayerNorm.weight"))], "weight_decay": 0.01},
        {"params": [p for name, p in model.named_parameters() if p.requires_grad and any(key in name for key in ("bias", "LayerNorm.bias", "LayerNorm.weight"))], "weight_decay": 0.0},
    ]
    optimizer = AdamW(grouped, lr=float(args.learning_rate))
    total_steps = len(train_loader) * args.num_train_epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup_epoch * len(train_loader)), total_steps)
    loss_fn = torch.nn.NLLLoss(weight=torch.tensor([1.0, float(args.class_weight)], device=args.device))
    best_f1, best_state, best_dev, best_epoch, history = -1.0, None, None, None, []

    for epoch in range(1, args.num_train_epoch + 1):
        model.train()
        losses = []
        for batch in train_loader:
            values = unpack(batch, args.device)
            loss = loss_fn(forward(model, values), values["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            losses.append(float(loss.item()))
        dev_metrics, _, _, _ = evaluate(model, dev_loader, args.device)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation": dev_metrics})
        print(json.dumps(history[-1]))
        if dev_metrics["f1"] > best_f1:
            best_f1, best_dev, best_epoch = dev_metrics["f1"], dev_metrics, epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("No checkpoint selected")
    model.load_state_dict(best_state)
    if cli.save_encoder:
        cli.save_encoder.parent.mkdir(parents=True, exist_ok=True)
        torch.save({name: value.detach().cpu() for name, value in model.encoder.state_dict().items()}, cli.save_encoder)
    test_metrics, labels, probability, prediction = evaluate(model, test_loader, args.device)
    cli.output_dir.mkdir(parents=True, exist_ok=False)
    payload = {
        "model": "RoPPT-Auto-spacy-sm", "seed": args.seed, "dataset": cli.dataset,
        "validation": best_dev, "test": test_metrics, "best_epoch": best_epoch,
        "protocol_note": "released RoPPT architecture; automatic spaCy-sm dependency parses; development-only checkpoint selection; test evaluated once",
        "epochs": args.num_train_epoch, "batch_size": args.per_gpu_train_batch_size,
        "class_weight": args.class_weight, "history": history,
        "encoder_init": str(cli.encoder_init.resolve()) if cli.encoder_init else None,
        "encoder_frozen": bool(cli.freeze_encoder),
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }
    (cli.output_dir / "strict_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savez_compressed(
        cli.output_dir / "test_predictions.npz",
        label=labels.astype(np.int8), probability=probability.astype(np.float32), prediction=prediction.astype(np.int8),
    )
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
