from __future__ import annotations
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "processing"))
from prepare_canonical_kfolds import read_rows, write_rows
from prepare_vua20_extension import main as prepare_vua20


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--misnet-data", type=Path, required=True)
    parser.add_argument("--vua20-source", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "data")
    cli = parser.parse_args()
    source = cli.misnet_data.resolve()
    output = cli.output.resolve()
    for name, folder, counts in (
        ("vua18", "VUA_All", (116622, 38628, 50175)),
        ("vua_verb_standard", "VUA_Verb", (15516, 1724, 5873)),
    ):
        for split, expected in zip(("train", "val", "test"), counts):
            rows = read_rows(source / folder / (split + ".csv"))
            if len(rows) != expected:
                raise ValueError(f"{name}/{split}: {len(rows)} != {expected}")
            write_rows(output / name / (split + ".csv"), rows)
    for name, relative, expected in (
        ("mohx", "MOH-X/MOH-X.csv", 647),
        ("trofi", "TroFi/TroFi.csv", 3737),
    ):
        if len(read_rows(source / relative)) != expected:
            raise ValueError(f"Unexpected {name} row count")
        subprocess.run([
            sys.executable, str(ROOT / "scripts/processing/prepare_canonical_kfolds.py"),
            "--source", str(source / relative), "--output", str(output / name),
            "--seed", "20260815", "--folds", "10",
        ], check=True)
    if cli.vua20_source:
        subprocess.run([
            sys.executable, str(ROOT / "scripts/processing/prepare_vua20_extension.py"),
            "--source", str(cli.vua20_source.resolve()),
            "--output-root", str(output / "vua20_build"),
            "--split-seed", "2026", "--validation-fraction", "0.0416",
        ], check=True)
        for split in ("train", "val", "test"):
            import shutil
            destination = output / "vua20" / (split + ".csv")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(output / "vua20_build/canonical" / (split + ".csv"), destination)
    subprocess.run([sys.executable, str(ROOT / "scripts/verify_resources.py"),
                    "--data-root", str(output)], check=True)


if __name__ == "__main__":
    main()
