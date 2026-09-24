"""Shared, deterministic feature contract for fraud-scoring training and serving.

This module intentionally performs no learned encoding. A future training
pipeline can fit an encoder on the stable output of this builder, then reuse
that fitted encoder during inference. Both paths must call the functions here
before reaching that preprocessing step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype, is_numeric_dtype

IDENTITY_COLUMN: Final[str] = "transaction_id"
TARGET_COLUMN: Final[str] = "is_fraud"

# This tuple is the single source of truth for estimator input order.
TRAINING_FEATURES: Final[tuple[str, ...]] = (
    "amount",
    "merchant_category",
    "hour_of_day",
    "device_risk",
)
SERVING_FEATURES: Final[tuple[str, ...]] = TRAINING_FEATURES


class FeatureContractError(ValueError):
    """Raised when raw input cannot produce safe, model-ready features."""


@dataclass(frozen=True)
class TrainingFeatures:
    """The ordered model matrix and training-only target produced from raw rows."""

    features: pd.DataFrame
    target: pd.Series


def build_training_features(transactions: pd.DataFrame) -> TrainingFeatures:
    """Build ordered model features and the target from validated historical rows."""
    _require_columns(transactions, (*TRAINING_FEATURES, IDENTITY_COLUMN, TARGET_COLUMN), "training")
    features = _build_features(transactions)
    target = _build_target(transactions[TARGET_COLUMN])
    return TrainingFeatures(features=features, target=target)


def build_serving_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Build ordered model features from authorization-time rows.

    Serving input must contain an identity and pre-authorization features, but
    must never include the training label.
    """
    if TARGET_COLUMN in transactions.columns:
        raise FeatureContractError(
            f"Serving input contains forbidden target column '{TARGET_COLUMN}'."
        )
    _require_columns(transactions, (*SERVING_FEATURES, IDENTITY_COLUMN), "serving")
    return _build_features(transactions)


def _require_columns(
    transactions: pd.DataFrame,
    required_columns: tuple[str, ...],
    mode: Literal["training", "serving"],
) -> None:
    if not isinstance(transactions, pd.DataFrame):
        raise FeatureContractError("Feature input must be a pandas DataFrame.")

    missing_columns = [column for column in required_columns if column not in transactions.columns]
    if missing_columns:
        raise FeatureContractError(
            f"{mode.capitalize()} input is missing required columns: {missing_columns}."
        )

    identity = transactions[IDENTITY_COLUMN]
    if identity.isna().any() or not identity.map(lambda value: isinstance(value, str)).all():
        raise FeatureContractError(f"'{IDENTITY_COLUMN}' must contain non-null string identities.")


def _build_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the fixed, non-leaking estimator input matrix."""
    amount = transactions["amount"]
    device_risk = transactions["device_risk"]
    hour_of_day = transactions["hour_of_day"]
    merchant_category = transactions["merchant_category"]

    _require_numeric(amount, "amount")
    _require_numeric(device_risk, "device_risk")
    if (
        hour_of_day.isna().any()
        or is_bool_dtype(hour_of_day)
        or not is_integer_dtype(hour_of_day)
    ):
        raise FeatureContractError("'hour_of_day' must use a non-null integer dtype.")
    if merchant_category.isna().any() or not merchant_category.map(
        lambda value: isinstance(value, str)
    ).all():
        raise FeatureContractError("'merchant_category' must contain non-null strings.")

    # Normalization is deliberate and fixed: every caller receives these
    # dtypes and this column order, irrespective of raw DataFrame ordering.
    return pd.DataFrame(
        {
            "amount": amount.astype("float64"),
            "merchant_category": merchant_category.astype("string"),
            "hour_of_day": hour_of_day.astype("int64"),
            "device_risk": device_risk.astype("float64"),
        },
        index=transactions.index,
    )


def _require_numeric(series: pd.Series, column_name: str) -> None:
    if series.isna().any() or is_bool_dtype(series) or not is_numeric_dtype(series):
        raise FeatureContractError(f"'{column_name}' must use a non-null numeric dtype.")


def _build_target(target: pd.Series) -> pd.Series:
    if target.isna().any() or is_bool_dtype(target) or not is_integer_dtype(target):
        raise FeatureContractError(f"'{TARGET_COLUMN}' must use a non-null integer dtype.")
    if not target.isin([0, 1]).all():
        raise FeatureContractError(f"'{TARGET_COLUMN}' must contain only 0 or 1.")
    return target.astype("int64").rename(TARGET_COLUMN)
