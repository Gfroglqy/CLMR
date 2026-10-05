from __future__ import annotations
import argparse
import csv
import re
from collections import Counter
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/processing"))
from prepare_melbert_crossval import convert_split
from prepare_vua_verb_standard import main as verb_main
from prepare_contrastwsd_data import build_lookup, convert_split as convert_wsd
from prepare_vua20_contrastwsd import main as vua20_wsd_main


def read(path, delimiter=","):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("vua18", "vua_verb_standard", "vua20", "mohx", "trofi"), required=True)
    parser.add_argument("--melbert-vua18", type=Path)
    parser.add_argument("--wsd-data", type=Path)
    parser.add_argument("--graphs", action="store_true")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data")
    cli = parser.parse_args()
    dataset = cli.dataset
    source = cli.data_root.resolve() / dataset
    output_root = cli.output_root.resolve()
    folds = [f"fold_{i:02d}" for i in range(10)] if dataset in ("mohx", "trofi") else [""]
    for fold in folds:
        csv_root = source / fold
        tsv_root = output_root / "baseline_inputs/melbert" / dataset / fold
        for split, tsv_split in (("train", "train"), ("val", "dev"), ("test", "test")):
            if dataset == "vua18":
                if cli.melbert_vua18 is None:
                    parser.error("VUA18 requires --melbert-vua18 with the authors' original train/dev/test TSVs")
                original = cli.melbert_vua18 / (tsv_split + ".tsv")
                canonical, supplied = read(csv_root / (split + ".csv")), read(original, "\t")
                keys = lambda row: (re.sub(r"[^a-z0-9]+", "", row["sentence"].lower()), int(row["label"]), int(row.get("target_position", row.get("w_index"))), row.get("pos_tag", row.get("POS")))
                if Counter(map(keys, canonical)) != Counter(map(keys, supplied)):
                    raise ValueError("MelBERT VUA18 row alignment failed; do not silently replace its source ids")
                tsv_root.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(original, tsv_root / (tsv_split + ".tsv"))
            else:
                convert_split(csv_root / (split + ".csv"), tsv_root / (tsv_split + ".tsv"), f"{dataset}-{fold}-{tsv_split}")
        if cli.graphs:
            import spacy
            from prepare_dags_data import read_csv, read_tsv, prepare_split
            nlp = spacy.load("en_core_web_sm", exclude=("ner", "lemmatizer"))
            for split, tsv_split in (("train", "train"), ("val", "dev"), ("test", "test")):
                rows = read_tsv(tsv_root / (tsv_split + ".tsv")) if dataset == "vua18" else read_csv(csv_root / (split + ".csv"))
                prepare_split(nlp, rows, output_root / "baseline_inputs/dags" / dataset / fold / (split + "_rospacy.json"))
    if cli.wsd_data is not None:
        if dataset in ("mohx", "trofi"):
            raise ValueError("No official contextual WSD input for MOH-X/TroFi; no aligned reproduction claimed")
        output = output_root / "baseline_inputs/contrastwsd" / dataset
        if dataset == "vua18":
            paths = [cli.wsd_data / folder / (split + ".tsv") for folder in ("VUA18", "VUA20") for split in ("train", "test")]
            exact, by_uid = build_lookup(paths)
            for split in ("train", "dev", "test"):
                convert_wsd(output_root / "baseline_inputs/melbert/vua18" / (split + ".tsv"), output / (split + ".tsv"), exact, by_uid)
        elif dataset == "vua_verb_standard":
            sys.argv = ["prepare_vua_verb_standard", "--source", str(source), "--official-wsd-root", str(cli.wsd_data), "--output-root", str(output_root / "verb_wsd_build")]
            verb_main()
            shutil.copytree(output_root / "verb_wsd_build/contrastwsd", output, dirs_exist_ok=True)
        else:
            sys.argv = ["prepare_vua20_contrastwsd", "--official-root", str(cli.wsd_data / "VUA20"), "--canonical-root", str(source), "--output-root", str(output)]
            vua20_wsd_main()


if __name__ == "__main__":
    main()
