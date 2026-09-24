"""Tests for the shared fraud-scoring feature builder."""

import pandas as pd
import pytest

from fraud_scoring.features import (
    IDENTITY_COLUMN,
    SERVING_FEATURES,
    TARGET_COLUMN,
    TRAINING_FEATURES,
    FeatureContractError,
    build_serving_features,
    build_training_features,
)


@pytest.fixture
def raw_training_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "device_risk": [0.20, 0.85],
            "is_fraud": [0, 1],
            "transaction_id": ["TXN_001", "TXN_002"],
            "merchant_category": ["grocery", "electronics"],
            "hour_of_day": [10, 2],
            "amount": [15.50, 420.00],
        }
    )


def test_training_builder_uses_fixed_feature_order(raw_training_transactions):
    result = build_training_features(raw_training_transactions)

    assert list(result.features.columns) == list(TRAINING_FEATURES)
    assert IDENTITY_COLUMN not in result.features.columns
    assert TARGET_COLUMN not in result.features.columns
    assert result.target.tolist() == [0, 1]


def test_builder_normalizes_model_feature_dtypes(raw_training_transactions):
    result = build_training_features(raw_training_transactions)

    assert str(result.features["amount"].dtype) == "float64"
    assert str(result.features["merchant_category"].dtype) == "string"
    assert str(result.features["hour_of_day"].dtype) == "int64"
    assert str(result.features["device_risk"].dtype) == "float64"
    assert str(result.target.dtype) == "int64"


def test_serving_builder_rejects_target(raw_training_transactions):
    with pytest.raises(FeatureContractError, match="forbidden target column"):
        build_serving_features(raw_training_transactions)


def test_builder_rejects_non_numeric_feature(raw_training_transactions):
    raw_training_transactions["amount"] = raw_training_transactions["amount"].astype(object)
    raw_training_transactions.loc[0, "amount"] = "15.50"

    with pytest.raises(FeatureContractError, match="numeric dtype"):
        build_training_features(raw_training_transactions)


def test_serving_contract_excludes_target_and_identity_from_model_features():
    assert TARGET_COLUMN not in SERVING_FEATURES
    assert TARGET_COLUMN not in TRAINING_FEATURES
    assert IDENTITY_COLUMN not in TRAINING_FEATURES
