"""Train/serve parity tests for the single shared feature builder."""

import pandas as pd

from fraud_scoring.features import (
    IDENTITY_COLUMN,
    SERVING_FEATURES,
    TARGET_COLUMN,
    TRAINING_FEATURES,
    build_serving_features,
    build_training_features,
)


def test_train_and_serve_emit_identical_features_for_same_raw_rows():
    training_rows = pd.DataFrame(
        {
            "transaction_id": ["TXN_100", "TXN_101"],
            "amount": [19.99, 250.00],
            "merchant_category": ["dining", "travel"],
            "hour_of_day": [12, 23],
            "device_risk": [0.12, 0.74],
            "is_fraud": [0, 1],
        }
    )
    serving_rows = training_rows.drop(columns=[TARGET_COLUMN])

    training_result = build_training_features(training_rows)
    serving_features = build_serving_features(serving_rows)

    assert list(training_result.features.columns) == list(TRAINING_FEATURES)
    assert list(serving_features.columns) == list(SERVING_FEATURES)
    assert list(training_result.features.columns) == list(serving_features.columns)
    assert training_result.features.dtypes.equals(serving_features.dtypes)
    pd.testing.assert_frame_equal(training_result.features, serving_features)


def test_identity_and_target_never_reach_model_matrix():
    training_rows = pd.DataFrame(
        {
            "transaction_id": ["TXN_200"],
            "amount": [75.00],
            "merchant_category": ["grocery"],
            "hour_of_day": [9],
            "device_risk": [0.25],
            "is_fraud": [0],
        }
    )

    model_features = build_training_features(training_rows).features

    assert IDENTITY_COLUMN not in model_features
    assert TARGET_COLUMN not in model_features
