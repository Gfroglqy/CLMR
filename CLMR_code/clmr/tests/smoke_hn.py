from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "src"))

from ark_dcr.data import (LiteralPrototypeMemory, PrototypeEvidenceCollator,
                          PrototypeEvidenceDataset, load_examples)
from model import LMRExpert


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=Path,
                        default=REPOSITORY / "pretrained" / "roberta-base")
    parser.add_argument("--data-dir", type=Path,
                        default=REPOSITORY / "data" / "vua18")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained, use_fast=True, add_prefix_space=True, local_files_only=True)
    raw = load_examples(args.data_dir / "train.csv")
    memory = LiteralPrototypeMemory(raw, 64, True)
    items = memory.materialize(raw[:96], 3, training=True)
    batch = next(iter(DataLoader(
        PrototypeEvidenceDataset(items), batch_size=96,
        collate_fn=PrototypeEvidenceCollator(tokenizer, 3, 150, 96))))
    batch = {key: value.to(device) if torch.is_tensor(value) else value
             for key, value in batch.items()}
    model = LMRExpert(str(args.pretrained), 384, 256, 0.1,
                      hard_negative_weight=1.0).to(device)
    model.set_training_progress(0.5)

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                        enabled=device.type == "cuda"):
        output = model(batch)
        loss = (torch.nn.functional.cross_entropy(output["logits"], batch["labels"])
                + 0.2 * output["auxiliary_loss"])
    loss.backward()
    print(json.dumps({
        "loss": float(loss.detach().cpu()),
        "matching_loss": float(output["matching_loss"].detach().cpu()),
        "finite": bool(torch.isfinite(loss).item()),
        "lemma_groups": int(batch["lemma_ids"].unique().numel()),
        "gradient_tensors": sum(parameter.grad is not None
                                for parameter in model.parameters()),
        "device": str(device),
    }, indent=2))


if __name__ == "__main__":
    main()
