
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from efficiency_utils import benchmark_cuda, collect_batches, model_size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--benchmark-batches", type=int, default=50)
    parser.add_argument("--warmup-batches", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    cli = parser.parse_args()

    repository = Path(__file__).resolve().parents[2]
    clmr_source = repository / "clmr" / "src"
    sys.path.insert(0, str(clmr_source))
    from ark_dcr.data import LiteralPrototypeMemory, PrototypeEvidenceCollator, PrototypeEvidenceDataset, load_examples
    from model import LMRExpert

    config_path = cli.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    resolve = lambda value: (config_path.parent / Path(value)).resolve()
    pretrained = resolve(config["pretrained_model"])
    data_dir = resolve(config["data_dir"])
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained, use_fast=True, add_prefix_space=True, local_files_only=True,
    )
    train = load_examples(data_dir / "train.csv")
    test = load_examples(data_dir / "test.csv")
    memory = LiteralPrototypeMemory(
        train, max_candidates=config["max_memory_candidates"], anchor_first=True,
    )
    items = memory.materialize(test, config["prototype_count"], training=False)
    loader = DataLoader(
        PrototypeEvidenceDataset(items), batch_size=cli.batch_size, shuffle=False,
        collate_fn=PrototypeEvidenceCollator(
            tokenizer, config["prototype_count"], config["max_context_length"],
            config["max_evidence_length"],
        ), num_workers=0, pin_memory=True,
    )
    device = torch.device("cuda")
    model = LMRExpert(
        pretrained, config["representation_size"], config["semantic_size"],
        config["dropout"], hard_negative_weight=config["hard_negative_weight"],
        hard_negative_temperature=config["hard_negative_temperature"],
    ).to(device)


    checkpoint = torch.load(cli.checkpoint.resolve(), map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    batches = collect_batches(loader, cli.benchmark_batches, device)
    result = {
        "model": "CLMR",
        "hardware": torch.cuda.get_device_name(0),
        "precision": "bf16",
        "batch_size": cli.batch_size,
        "max_context_length": config["max_context_length"],
        "max_evidence_length": config["max_evidence_length"],
        "prototype_count": config["prototype_count"],
        **model_size(model),
        **benchmark_cuda(
            model, batches, lambda current, batch: current(batch),
            warmup_batches=cli.warmup_batches, repeats=cli.repeats,
        ),
    }
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
