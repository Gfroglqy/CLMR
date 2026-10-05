from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", nargs="+", choices=("melbert", "dags", "roppt", "contrastwsd", "misnet"))
    parser.add_argument("--backbone", action="store_true")
    parser.add_argument("--accept-upstream-terms", action="store_true")
    cli = parser.parse_args()
    if not cli.accept_upstream_terms:
        parser.error("Read the upstream resource terms, then pass --accept-upstream-terms")
    if not cli.upstream and not cli.backbone:
        parser.error("Choose --upstream and/or --backbone")
    sources = json.loads((ROOT / "resources/upstream_sources.json").read_text())
    git = ["git"] + (["-c", "http.sslBackend=openssl"] if sys.platform == "win32" else [])
    for name in cli.upstream or []:
        item = sources[name]
        destination = ROOT / "external" / name
        if destination.exists():
            actual = subprocess.check_output(git + ["-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
            if actual != item["commit"]:
                raise RuntimeError(f"Wrong upstream revision at {destination}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(git + ["clone", item["url"], str(destination)], check=True)
            subprocess.run(git + ["-C", str(destination), "checkout", "--detach", item["commit"]], check=True)
    if cli.backbone:
        from huggingface_hub import snapshot_download
        location = snapshot_download("FacebookAI/roberta-base", local_dir=ROOT / "pretrained/roberta-base",
            allow_patterns=["config.json", "vocab.json", "merges.txt", "tokenizer.json", "model.safetensors"])
        print("Backbone downloaded to", location)
    print("Third-party assets remain in ignored directories, outside the distributed source files.")


if __name__ == "__main__":
    main()
