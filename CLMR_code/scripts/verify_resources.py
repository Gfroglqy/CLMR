from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--require-all", action="store_true")
    cli = parser.parse_args()
    manifest = json.loads((ROOT / "resources/data_manifest.json").read_text())
    missing, verified = [], 0
    for relative, expected in manifest["files"].items():
        path = cli.data_root / relative
        if not path.exists():
            missing.append(relative)
            continue
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        body = [[row.get(key, "") for key in manifest["fields"]] for row in rows]
        actual = hashlib.sha256(json.dumps(body, ensure_ascii=False,
            separators=(",", ":")).encode("utf-8")).hexdigest()
        if len(rows) != expected["instances"] or actual != expected["content_sha256"]:
            raise ValueError(f"Frozen input mismatch: {relative}")
        verified += 1
    if verified == 0 or (cli.require_all and missing):
        raise FileNotFoundError(f"Missing frozen data: {missing}")
    print(json.dumps({"verified_splits": verified, "missing_splits": missing}, indent=2))


if __name__ == "__main__":
    main()
