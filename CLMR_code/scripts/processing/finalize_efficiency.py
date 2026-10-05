
from __future__ import annotations

import csv
import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "formal" / "efficiency"
ORDER = ["CLMR", "MelBERT", "DAGS-Auto-spacy-sm", "RoPPT-Auto-spacy-sm", "ContrastWSD-Aligned"]


def main() -> None:
    global REPORT
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True,
                        help="Directory containing one benchmark JSON per model")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    REPORT = args.input.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    files = list(REPORT.glob("*.json"))
    rows = []
    for path in files:
        value = json.loads(path.read_text(encoding="utf-8"))
        if "instances_per_second_mean" not in value:
            continue
        rows.append(value)
    by_name = {row["model"]: row for row in rows}
    missing = [name for name in ORDER if name not in by_name]
    if missing:
        raise RuntimeError(f"Missing efficiency results: {missing}")
    rows = [by_name[name] for name in ORDER]

    fields = [
        "model", "total_parameters", "batch_size", "instances_per_second_mean",
        "instances_per_second_std", "milliseconds_per_instance_mean",
        "milliseconds_per_instance_std", "peak_gpu_gb_mean", "peak_gpu_gb_std",
    ]
    csv_path = output / "formal_efficiency.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)

    lines = [
        "# Formal same-device inference efficiency", "",
        "RTX 4080 SUPER; BF16; batch size 32; VUA-All test inputs; "
        "10 warm-up batches and 50 timed batches, repeated three times. "
        "Data preparation and checkpoint loading are excluded.", "",
        "| Model | Parameters | targets/s | ms/target | Peak GPU |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['total_parameters']/1e6:.1f}M | "
            f"{row['instances_per_second_mean']:.1f} +/- {row['instances_per_second_std']:.1f} | "
            f"{row['milliseconds_per_instance_mean']:.3f} +/- {row['milliseconds_per_instance_std']:.3f} | "
            f"{row['peak_gpu_gb_mean']:.2f} +/- {row['peak_gpu_gb_std']:.2f} GB |"
        )
    (output / "formal_efficiency.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
