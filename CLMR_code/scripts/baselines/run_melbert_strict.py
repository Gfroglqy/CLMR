from __future__ import annotations
import argparse
import importlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, brier_score_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "processing"))
from aggregate_crossval import expected_calibration_error


def main():
    parser = argparse.ArgumentParser()
    for name in ("repo", "backbone", "data-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--class-weight", type=float, required=True)
    parser.add_argument("--encoder-init", type=Path)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--save-encoder", type=Path)
    parser.add_argument("--check-only", action="store_true")
    cli = parser.parse_args()
    sys.path.insert(0, str(cli.repo.resolve()))
    upstream = importlib.import_module("main")
    from utils import Config, Logger
    from run_classifier_dataset_utils import processors, output_modes
    args = Config(str(cli.repo.resolve()))
    args.bert_model = str(cli.backbone.resolve())
    args.data_dir = str(cli.data_dir.resolve())
    args.task_name, args.model_type = "vua", "MELBERT"
    args.num_train_epoch, args.train_batch_size = cli.epochs, cli.batch_size
    args.eval_batch_size, args.max_seq_length = 96, 150
    args.learning_rate, args.class_weight, args.warmup_epoch = 3e-5, cli.class_weight, 2
    args.seed, args.num_bagging, args.num_labels = cli.seed, 0, 2
    args.do_train, args.do_eval, args.do_test = True, True, True
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = 1 if torch.cuda.is_available() else 0
    random.seed(cli.seed); np.random.seed(cli.seed); torch.manual_seed(cli.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(cli.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    cli.output_dir.mkdir(parents=True, exist_ok=False)
    args.log_dir = str(cli.output_dir.resolve())
    logger = Logger(args.log_dir)
    processor = processors["vua"]()
    labels, mode = processor.get_labels(), output_modes["vua"]
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    train = upstream.load_train_data(args, logger, processor, "vua", labels, tokenizer, mode)
    if cli.check_only:
        print(json.dumps({"train": len(train.dataset), "dev": len(processor.get_dev_examples(args.data_dir)), "test": len(processor.get_test_examples(args.data_dir))}))
        return
    model = upstream.load_pretrained_model(args)
    if cli.encoder_init:
        model.encoder.load_state_dict(torch.load(cli.encoder_init, map_location="cpu", weights_only=True), strict=True)
    if cli.freeze_encoder:
        for parameter in model.encoder.parameters(): parameter.requires_grad = False
    best = {"f1": -1., "state": None, "epoch": 0}
    original_loader = upstream.load_test_data

    def dev_loader(*values, **kwargs):
        original_test = processor.get_test_examples
        processor.get_test_examples = processor.get_dev_examples
        try:
            return original_loader(*values, **kwargs)
        finally:
            processor.get_test_examples = original_test

    def evaluate(current, loader):
        current.eval()
        ys, ps = [], []
        with torch.inference_mode():
            for batch in loader:
                ids, mask, seg, y, ids2, mask2, seg2 = [value.to(args.device) for value in batch]
                logits = current(ids, ids2, target_mask=(seg == 1), target_mask_2=seg2,
                    attention_mask_2=mask2, token_type_ids=seg, attention_mask=mask)
                ys.append(y.cpu().numpy()); ps.append(torch.softmax(logits, -1)[:, 1].cpu().numpy())
        y, p = np.concatenate(ys).astype(np.int64), np.concatenate(ps).astype(np.float64)
        pred = (p >= .5).astype(np.int64)
        scores = {"accuracy": float(accuracy_score(y, pred)), "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)), "f1": float(f1_score(y, pred, zero_division=0)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "nll": float(log_loss(y, np.column_stack((1-p, p)), labels=[0,1])),
            "brier": float(brier_score_loss(y, p)), "ece": expected_calibration_error(y, p)}
        return scores, y, p, pred

    def dev_eval(args_, logger_, current, loader, guids, task_name, **kwargs):
        scores, _, _, _ = evaluate(current, loader)
        best["epoch"] += 1
        if scores["f1"] > best["f1"]:
            best.update(f1=scores["f1"], state={k:v.detach().cpu().clone() for k,v in current.state_dict().items()},
                        metrics=scores, best_epoch=best["epoch"])
        print(json.dumps({"epoch": best["epoch"], "validation": scores}))
        return scores

    upstream.load_test_data = dev_loader
    upstream.run_eval = dev_eval
    upstream.save_model = lambda *values, **kwargs: None
    model, _ = upstream.run_train(args, logger, model, train, processor, "vua", labels, tokenizer, mode)
    if best["state"] is None: raise RuntimeError("No validation checkpoint")
    model.load_state_dict(best["state"], strict=True)
    if cli.save_encoder:
        cli.save_encoder.parent.mkdir(parents=True, exist_ok=True)
        torch.save({k:v.detach().cpu() for k,v in model.encoder.state_dict().items()}, cli.save_encoder)
    guids, test_loader = original_loader(args, logger, processor, "vua", labels, tokenizer, mode)
    test, y, p, pred = evaluate(model, test_loader)
    payload = {"model":"MelBERT", "seed":cli.seed, "dataset":cli.dataset,
        "validation":best["metrics"], "test":test, "best_epoch":best["best_epoch"],
        "epochs":cli.epochs, "batch_size":cli.batch_size, "class_weight":cli.class_weight,
        "encoder_frozen":cli.freeze_encoder, "protocol":"validation-only selection; one final outer-test evaluation"}
    (cli.output_dir / "strict_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savez_compressed(cli.output_dir / "test_predictions.npz", guid=np.asarray(guids).astype(str),
                        label=y.astype(np.int8), probability=p.astype(np.float32), prediction=pred.astype(np.int8))
    print(json.dumps(test, indent=2))


if __name__ == "__main__":
    main()
