"""Tests for the DVC preparation stage."""

import pandas as pd

from fraud_scoring.generate_data import generate_synthetic_transactions
from fraud_scoring.prepare import prepare_dataset


def test_prepare_validates_and_writes_deterministic_dataset(tmp_path):
    input_path = tmp_path / "transactions.csv"
    output_path = tmp_path / "processed" / "validated.csv"

    source = generate_synthetic_transactions(
        n_records=200,
        fraud_ratio=0.04,
        random_seed=42,
    )

    source.to_csv(input_path, index=False)

    prepare_dataset(
        input_path,
        output_path,
    )

    prepared = pd.read_csv(output_path)

    pd.testing.assert_frame_equal(
        source,
        prepared,
    )
