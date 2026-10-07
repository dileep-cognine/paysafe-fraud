"""Prepare the raw transaction dataset for the reproducible DVC pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fraud_scoring.data_validation import validate_and_enforce


def prepare_dataset(input_path: Path, output_path: Path) -> None:
    """Validate and deterministically normalize the raw training dataset."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    raw = pd.read_csv(input_path)
    validated = validate_and_enforce(raw, is_training=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    validated.to_csv(output_path, index=False)

    print(f"Prepared {len(validated)} validated rows -> {output_path}")


def main() -> None:
    """Prepare a validated dataset from command-line input and output paths."""
    parser = argparse.ArgumentParser(description="Prepare validated fraud-scoring training data")

    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/raw/transactions.csv"),
        help="Path to the DVC-tracked raw CSV",
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/processed/validated.csv"),
        help="Path for the validated CSV produced by this stage",
    )

    args = parser.parse_args()

    prepare_dataset(
        input_path=args.input_path,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
