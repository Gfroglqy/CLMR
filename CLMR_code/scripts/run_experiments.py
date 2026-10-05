from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = {"vua18":"vua_all", "vua_verb_standard":"vua_verb_standard", "vua20":"vua20_extension", "mohx":"mohx", "trofi":"trofi"}
CONFIGS = {"vua18":"clmr_vua18.json", "vua_verb_standard":"clmr_vua_verb.json", "vua20":"clmr_vua20.json", "mohx":"mohx_e2e_clmr.json", "trofi":"trofi_e2e_clmr.json"}


def execute(args):
    print("RUN:", " ".join(str(x) for x in args), flush=True)
    subprocess.run([sys.executable, *map(str, args)], check=True, cwd=ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("clmr", "melbert", "dags", "roppt", "contrastwsd"), required=True)
    parser.add_argument("--dataset", choices=tuple(NAMES), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[13,21,42])
    parser.add_argument("--mode", choices=("train", "frozen"), default="train")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts/runs")
    parser.add_argument("--check-only", action="store_true")
    cli = parser.parse_args()
    dataset, model = cli.dataset, cli.model
    cv = dataset in ("mohx", "trofi")
    if cli.mode == "frozen" and not cv:
        parser.error("Frozen transfer is a separate MOH-X/TroFi protocol")
    if model == "contrastwsd" and cv:
        parser.error("Official WSD augmentation does not cover MOH-X/TroFi; this table cell is unavailable")
    for seed in cli.seeds:
        if seed not in (13,21,42): parser.error("Frozen paper protocol uses seeds 13,21,42")
        source = None
        if cli.mode == "frozen":
            if cli.source_root is not None:
                source = cli.source_root / f"seed_{seed}" / ("checkpoints/best.pt" if model == "clmr" else "encoder.pt")
                if not source.exists(): raise FileNotFoundError(source)
            else:
                source_run = cli.output_root / "source_training" / model / f"seed_{seed}"
                source = source_run / ("checkpoints/best.pt" if model == "clmr" else "encoder.pt")
                if not source.exists():
                    if model == "clmr":
                        execute([ROOT / "clmr/src/train.py", "--config", ROOT / "clmr/configs/clmr_transfer_source.json", "--seed", seed, "--output", source_run])
                    else:
                        base = [ROOT / f"scripts/baselines/run_{model}_strict.py", "--repo", ROOT / "external" / model,
                            "--backbone", ROOT / "pretrained/roberta-base", "--data-dir", ROOT / "data/baseline_inputs" / ("melbert" if model=="melbert" else "dags") / "vua18",
                            "--output-dir", source_run, "--dataset", "vua_all", "--seed", seed,
                            "--epochs", 3 if model=="melbert" else 10, "--batch-size", 32 if model=="melbert" else 20,
                            "--class-weight", 3 if model=="melbert" else 5, "--save-encoder", source]
                        if model == "roppt": base += ["--dependency-repo", ROOT / "external/dags"]
                        execute(base)
        for fold in ([f"fold_{i:02d}" for i in range(10)] if cv else [""]):
            output = cli.output_root / cli.mode / model / dataset / f"seed_{seed}" / fold
            if (output / "metrics.json").exists() or (output / "strict_metrics.json").exists():
                raise FileExistsError(f"Refusing to overwrite completed run: {output}")
            if model == "clmr":
                config = "clmr_frozen_target.json" if cli.mode == "frozen" else CONFIGS[dataset]
                command = [ROOT / "clmr/src/train.py", "--config", ROOT / "clmr/configs" / config,
                    "--data-dir", ROOT / "data" / dataset / fold, "--seed", seed, "--output", output]
                command += ["--evaluate-test"]
                if cli.mode == "frozen":
                    command += ["--initial-checkpoint", source, "--freeze-context-encoder", "--freeze-knowledge-encoder"]
                if cli.check_only: parser.error("CLMR --check-only is not provided; use scripts/verify_resources.py or clmr/tests/smoke_hn.py")
            else:
                graph = model in ("dags", "roppt")
                input_kind = "dags" if graph else model
                epochs = 10 if graph or model=="contrastwsd" else (8 if dataset=="mohx" else 4 if dataset=="trofi" else 3)
                batch = 20 if graph or model=="contrastwsd" else (16 if dataset=="mohx" else 32)
                weight = 1 if cv else 5 if dataset in ("vua18", "vua20") else 3
                command = [ROOT / f"scripts/baselines/run_{model}_strict.py", "--repo", ROOT / "external" / model,
                    "--backbone", ROOT / "pretrained/roberta-base", "--data-dir", ROOT / "data/baseline_inputs" / input_kind / dataset / fold,
                    "--output-dir", output, "--dataset", NAMES[dataset], "--seed", seed,
                    "--epochs", epochs, "--batch-size", batch, "--class-weight", weight]
                if model == "roppt": command += ["--dependency-repo", ROOT / "external/dags"]
                if cli.mode == "frozen": command += ["--encoder-init", source, "--freeze-encoder"]
                if cli.check_only: command += ["--check-only"]
            execute(command)
        if not cli.check_only:
            execute([ROOT / "scripts/summarize_runs.py", "--run-root", cli.output_root / cli.mode / model / dataset,
                "--seeds", *cli.seeds, "--allow-incomplete"])


if __name__ == "__main__":
    main()
