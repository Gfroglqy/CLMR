"""Strict trainer for the released ContrastWSD architecture.

Checkpoint selection uses validation F1 only.  The frozen test split is
evaluated once, and no model checkpoint is persisted.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "processing"))

import numpy as np
import torch
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss, precision_score, recall_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup


def ece(labels: np.ndarray, probability: np.ndarray, bins: int = 15) -> float:
    prediction = (probability >= 0.5).astype(np.int64)
    confidence = np.where(prediction == 1, probability, 1.0 - probability)
    correct = (prediction == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index in range(bins):
        lower = confidence >= edges[index] if index == 0 else confidence > edges[index]
        selected = lower & (confidence <= edges[index + 1])
        if np.any(selected):
            value += np.mean(selected) * abs(np.mean(correct[selected]) - np.mean(confidence[selected]))
    return float(value)


def tensor_dataset(examples, converter, tokenizer, args) -> TensorDataset:
    features = converter(examples, ["0", "1"], args.max_seq_length, tokenizer, "classification", args)
    if len(features) != len(examples):
        raise RuntimeError(f"Feature conversion dropped rows: examples={len(examples)} features={len(features)}")
    return TensorDataset(
        torch.tensor([f.input_ids for f in features], dtype=torch.long),
        torch.tensor([f.input_mask for f in features], dtype=torch.long),
        torch.tensor([f.segment_ids for f in features], dtype=torch.long),
        torch.tensor([f.label_id for f in features], dtype=torch.long),
        torch.tensor([f.input_ids_2 for f in features], dtype=torch.long),
        torch.tensor([f.input_mask_2 for f in features], dtype=torch.long),
        torch.tensor([f.segment_ids_2 for f in features], dtype=torch.long),
        torch.tensor([f.input_ids_3 for f in features], dtype=torch.long),
        torch.tensor([f.input_mask_3 for f in features], dtype=torch.long),
        torch.tensor([f.segment_ids_3 for f in features], dtype=torch.long),
    )


def unpack(batch, device: torch.device) -> dict[str, torch.Tensor]:
    values = tuple(item.to(device) for item in batch)
    return {
        "input_ids": values[0], "attention_mask": values[1], "segment_ids": values[2],
        "labels": values[3], "input_ids_2": values[4], "attention_mask_2": values[5],
        "segment_ids_2": values[6], "input_ids_3": values[7],
        "attention_mask_3": values[8], "segment_ids_3": values[9],
    }


def forward(model: torch.nn.Module, values: dict[str, torch.Tensor]) -> torch.Tensor:
    return model(
        values["input_ids"], values["input_ids_2"],
        target_mask=(values["segment_ids"] == 1), target_mask_2=values["segment_ids_2"],
        attention_mask_2=values["attention_mask_2"], token_type_ids=values["segment_ids"],
        attention_mask=values["attention_mask"], input_ids_3=values["input_ids_3"],
        input_mask_3=values["attention_mask_3"], segment_ids_3=values["segment_ids_3"],
    )


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    labels, probabilities = [], []
    with torch.no_grad():
        for batch in loader:
            values = unpack(batch, device)
            log_probability = forward(model, values)
            labels.append(values["labels"].cpu().numpy())
            probabilities.append(torch.exp(log_probability)[:, 1].cpu().numpy())
    y = np.concatenate(labels).astype(np.int64)
    p = np.concatenate(probabilities).astype(np.float64)
    prediction = (p >= 0.5).astype(np.int64)
    metrics = {
        "accuracy": float(accuracy_score(y, prediction)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "nll": float(log_loss(y, np.column_stack((1.0 - p, p)), labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "ece": ece(y, p),
    }
    return metrics, y, p, prediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        choices=("vua_all", "vua_verb", "vua_verb_standard", "vua20_extension"),
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--class-weight", type=float, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--forward-check", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--benchmark-output", type=Path)
    parser.add_argument("--benchmark-batches", type=int, default=50)
    parser.add_argument("--warmup-batches", type=int, default=10)
    parser.add_argument("--benchmark-repeats", type=int, default=3)
    cli = parser.parse_args()

    sys.path.insert(0, str(cli.repo.resolve()))
    from utils import Config
    from main import load_pretrained_model
    from run_classifier_dataset_utils import VUAProcessor, convert_examples_to_two_features_with_wsd

    args = Config(str(cli.repo.resolve()))
    args.data_dir = str(cli.data_dir.resolve())
    args.bert_model = str(cli.backbone.resolve())
    args.model_type = "CONTRAST"
    args.seed = cli.seed
    args.num_train_epoch = cli.epochs
    args.train_batch_size = cli.batch_size
    args.eval_batch_size = max(cli.batch_size, 32)
    args.class_weight = cli.class_weight
    args.num_labels = 2
    selected_device = ("cuda" if torch.cuda.is_available() else "cpu") if cli.device == "auto" else cli.device
    args.device = torch.device(selected_device)
    args.n_gpu = 1 if selected_device == "cuda" else 0

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tokenizer = AutoTokenizer.from_pretrained(args.bert_model, do_lower_case=False)
    processor = VUAProcessor()
    train = tensor_dataset(processor.get_train_examples(args.data_dir), convert_examples_to_two_features_with_wsd, tokenizer, args)
    dev = tensor_dataset(processor.get_dev_examples(args.data_dir), convert_examples_to_two_features_with_wsd, tokenizer, args)
    test = tensor_dataset(processor.get_test_examples(args.data_dir), convert_examples_to_two_features_with_wsd, tokenizer, args)
    train_loader = DataLoader(train, sampler=RandomSampler(train), batch_size=args.train_batch_size)
    dev_loader = DataLoader(dev, sampler=SequentialSampler(dev), batch_size=args.eval_batch_size)
    test_loader = DataLoader(test, sampler=SequentialSampler(test), batch_size=args.eval_batch_size)

    if cli.check_only:
        batch = next(iter(train_loader))
        payload = {
            "train": len(train), "validation": len(dev), "test": len(test),
            "batch_fields": len(batch), "batch_size": int(batch[0].shape[0]),
            "shapes": [list(item.shape) for item in batch],
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
            "model": "ContrastWSD-Aligned",
            "hardware": torch.cuda.get_device_name(0),
            "precision": "bf16",
            "batch_size": int(args.eval_batch_size),
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
    grouped = [
        {"params": [p for name, p in model.named_parameters() if not any(key in name for key in ("bias", "LayerNorm.bias", "LayerNorm.weight"))], "weight_decay": 0.01},
        {"params": [p for name, p in model.named_parameters() if any(key in name for key in ("bias", "LayerNorm.bias", "LayerNorm.weight"))], "weight_decay": 0.0},
    ]
    optimizer = AdamW(grouped, lr=float(args.learning_rate))
    total_steps = len(train_loader) * args.num_train_epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup_epoch * len(train_loader)), total_steps)
    loss_fn = torch.nn.NLLLoss(weight=torch.tensor([1.0, float(args.class_weight)], device=args.device))
    best_f1, best_state, best_dev, best_epoch, history = -1.0, None, None, None, []
    progress_path = cli.output_dir.parent / f".{cli.output_dir.name}.progress.json"

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
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text(json.dumps({
            "model": "ContrastWSD-Aligned", "dataset": cli.dataset,
            "seed": args.seed, "completed_epoch": epoch,
            "total_epochs": args.num_train_epoch, "best_epoch": best_epoch,
            "best_validation_f1": best_f1, "latest": history[-1],
        }, indent=2), encoding="utf-8")

    if best_state is None:
        raise RuntimeError("No checkpoint selected")
    model.load_state_dict(best_state)
    test_metrics, labels, probability, prediction = evaluate(model, test_loader, args.device)
    cli.output_dir.mkdir(parents=True, exist_ok=False)
    payload = {
        "model": "ContrastWSD-Aligned", "seed": args.seed, "dataset": cli.dataset,
        "validation": best_dev, "test": test_metrics, "best_epoch": best_epoch,
        "protocol_note": "released ContrastWSD architecture and authors' WSD augmentation aligned to frozen strict VUA rows; missing released matches use the paper's target-token fallback",
        "epochs": args.num_train_epoch, "batch_size": args.train_batch_size,
        "class_weight": args.class_weight, "history": history,
    }
    (cli.output_dir / "strict_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savez_compressed(
        cli.output_dir / "test_predictions.npz",
        label=labels.astype(np.int8), probability=probability.astype(np.float32), prediction=prediction.astype(np.int8),
    )
    progress_path.unlink(missing_ok=True)
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
