"""Unit tests for schema validation, data quality checks, and leakage guardrails."""

import pandas as pd
import pytest

from fraud_scoring.data_validation import (
    IDENTITY_COLUMN,
    SERVING_FEATURES,
    TARGET,
    TRAIN_FEATURES,
    DataLeakageError,
    DataValidationError,
    assert_contract_invariants,
    assert_no_leakage,
    check_leakage,
    validate_and_enforce,
    validate_raw_transactions,
)


@pytest.fixture
def valid_train_df() -> pd.DataFrame:
    """Fixture providing a known-valid training DataFrame."""
    return pd.DataFrame(
        {
            "transaction_id": ["TXN_001", "TXN_002", "TXN_003"],
            "amount": [10.50, 250.00, 45.20],
            "merchant_category": ["grocery", "electronics", "fashion"],
            "hour_of_day": [10, 23, 0],
            "device_risk": [0.12, 0.85, 0.40],
            "is_fraud": [0, 1, 0],
        }
    )


@pytest.fixture
def valid_serve_df(valid_train_df: pd.DataFrame) -> pd.DataFrame:
    """Fixture providing a known-valid serving DataFrame (no target)."""
    return valid_train_df.drop(columns=["is_fraud"])


# =====================================================================
# 1. Happy Path Tests
# =====================================================================


def test_valid_training_dataset_passes(valid_train_df):
    result = validate_raw_transactions(valid_train_df, is_training=True)
    assert result.is_valid is True
    assert len(result.errors) == 0
    assert result.total_records == 3
    assert result.quality_summary["fraud_rate"] == pytest.approx(1 / 3)


def test_valid_serving_dataset_passes(valid_serve_df):
    result = validate_raw_transactions(valid_serve_df, is_training=False)
    assert result.is_valid is True
    assert len(result.errors) == 0
    assert result.total_records == 3


def test_validate_and_enforce_returns_df_when_valid(valid_train_df):
    validated = validate_and_enforce(valid_train_df, is_training=True)
    assert len(validated) == len(valid_train_df)


# =====================================================================
# 2. Data Quality & Schema Failure Tests
# =====================================================================


def test_missing_values_fail(valid_train_df):
    df = valid_train_df.copy()
    df.loc[0, "amount"] = None
    result = validate_raw_transactions(df, is_training=True)
    assert result.is_valid is False
    assert any("amount" in err and "missing" in err for err in result.errors)

    with pytest.raises(DataValidationError, match="missing"):
        validate_and_enforce(df, is_training=True)


def test_invalid_amount_fails(valid_train_df):
    # Non-positive amount (0 or negative)
    df_zero = valid_train_df.copy()
    df_zero.loc[1, "amount"] = 0.0
    result_zero = validate_raw_transactions(df_zero, is_training=True)
    assert result_zero.is_valid is False
    assert any("amount" in err and "non-positive" in err for err in result_zero.errors)

    df_neg = valid_train_df.copy()
    df_neg.loc[2, "amount"] = -15.50
    result_neg = validate_raw_transactions(df_neg, is_training=True)
    assert result_neg.is_valid is False
    assert any("amount" in err for err in result_neg.errors)


def test_numeric_strings_are_not_silently_coerced(valid_train_df):
    """The input contract must reject strings where a numeric value is required."""
    df = valid_train_df.copy()
    df["amount"] = df["amount"].astype(object)
    df.loc[0, "amount"] = "10.50"

    result = validate_raw_transactions(df, is_training=True)

    assert result.is_valid is False
    assert any("Pandera schema check failed" in err for err in result.errors)


def test_invalid_hour_fails(valid_train_df):
    df = valid_train_df.copy()
    df.loc[0, "hour_of_day"] = 24  # Valid hours are 0..23
    result = validate_raw_transactions(df, is_training=True)
    assert result.is_valid is False
    assert any("hour_of_day" in err for err in result.errors)

    df.loc[0, "hour_of_day"] = -1
    result_neg = validate_raw_transactions(df, is_training=True)
    assert result_neg.is_valid is False
    assert any("hour_of_day" in err for err in result_neg.errors)


def test_invalid_device_risk_fails(valid_train_df):
    df = valid_train_df.copy()
    df.loc[0, "device_risk"] = 1.25  # Risk must be in [0.0, 1.0]
    result = validate_raw_transactions(df, is_training=True)
    assert result.is_valid is False
    assert any("device_risk" in err for err in result.errors)

    df.loc[0, "device_risk"] = -0.05
    result_neg = validate_raw_transactions(df, is_training=True)
    assert result_neg.is_valid is False
    assert any("device_risk" in err for err in result_neg.errors)


def test_duplicate_transaction_ids_fail(valid_train_df):
    df = valid_train_df.copy()
    df.loc[1, "transaction_id"] = "TXN_001"  # Duplicate of row 0
    result = validate_raw_transactions(df, is_training=True)
    assert result.is_valid is False
    assert any("duplicate" in err for err in result.errors)


def test_invalid_merchant_category_fails(valid_train_df):
    df = valid_train_df.copy()
    df.loc[0, "merchant_category"] = "illicit_contraband"  # Not in allowed categories
    result = validate_raw_transactions(df, is_training=True)
    assert result.is_valid is False
    assert any("merchant_category" in err and "unauthorized" in err for err in result.errors)


def test_invalid_target_values_fail(valid_train_df):
    df = valid_train_df.copy()
    df.loc[0, "is_fraud"] = 5  # Must be 0 or 1
    result = validate_raw_transactions(df, is_training=True)
    assert result.is_valid is False
    assert any("is_fraud" in err and "invalid target" in err for err in result.errors)


def test_unexpected_columns_fail_strict_schema(valid_train_df):
    df = valid_train_df.copy()
    df["extra_unapproved_column"] = 123
    result = validate_raw_transactions(df, is_training=True)
    assert result.is_valid is False
    assert any("Unexpected columns detected" in err for err in result.errors)


def test_empty_dataset_fails():
    empty_df = pd.DataFrame(
        columns=[
            "transaction_id",
            "amount",
            "merchant_category",
            "hour_of_day",
            "device_risk",
            "is_fraud",
        ]
    )
    result = validate_raw_transactions(empty_df, is_training=True)
    assert result.is_valid is False
    assert any("empty" in err for err in result.errors)


# =====================================================================
# 3. Leakage Protection & Invariant Tests
# =====================================================================


def test_serving_features_do_not_contain_target():
    """Verify that is_fraud is never present in SERVING_FEATURES."""
    assert TARGET not in SERVING_FEATURES
    assert "is_fraud" not in SERVING_FEATURES
    assert "fraud_label" not in SERVING_FEATURES


def test_train_features_do_not_contain_target_or_id():
    """Verify that neither target nor transaction ID are in TRAIN_FEATURES."""
    assert TARGET not in TRAIN_FEATURES
    assert IDENTITY_COLUMN not in TRAIN_FEATURES


def test_target_leakage_detected_in_feature_columns():
    """Assert_no_leakage must raise DataLeakageError if is_fraud appears in features."""
    bad_features = ["amount", "hour_of_day", "is_fraud"]
    violations = check_leakage(bad_features, is_serving=False)
    assert len(violations) > 0
    assert any("is_fraud" in v for v in violations)

    with pytest.raises(DataLeakageError, match="critical target leakage"):
        assert_no_leakage(bad_features, is_serving=False)


def test_target_derived_feature_is_rejected():
    """A feature calculated from the label must be treated as leakage."""
    with pytest.raises(DataLeakageError, match="Target-derived columns detected"):
        assert_no_leakage(["amount", "target_rolling_fraud_rate"])


def test_target_leakage_in_serving_dataset_fails(valid_train_df):
    """If serving dataset inadvertently includes is_fraud, it must be rejected immediately."""
    result = validate_raw_transactions(valid_train_df, is_training=False)
    assert result.is_valid is False
    assert any("is_fraud" in err and "leakage" in err for err in result.errors)


def test_identity_leakage_detected_in_model_features():
    """Transaction ID must not be fed to model estimator."""
    bad_features = ["amount", "hour_of_day", "transaction_id"]
    with pytest.raises(DataLeakageError, match="Identity column 'transaction_id' detected"):
        assert_no_leakage(bad_features)


def test_post_authorization_features_rejected():
    """Post-auth fields such as chargeback_amount must trigger leakage error."""
    bad_features = ["amount", "device_risk", "chargeback_amount"]
    with pytest.raises(DataLeakageError, match="Post-authorization columns detected"):
        assert_no_leakage(bad_features)


def test_contract_invariants():
    """Verify system invariants check succeeds."""
    assert_contract_invariants()
