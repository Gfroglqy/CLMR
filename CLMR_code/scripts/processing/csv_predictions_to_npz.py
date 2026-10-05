
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    required = {"label", "probability"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns {required - set(frame.columns)} in {args.input}")
    labels = frame["label"].to_numpy(dtype=np.int8)
    probability = frame["probability"].to_numpy(dtype=np.float32)
    np.savez_compressed(
        args.output,
        label=labels,
        probability=probability,
        prediction=(probability >= 0.5).astype(np.int8),
    )


if __name__ == "__main__":
    main()
