"""Data validation, schema enforcement, and leakage protection module.

Ensures raw data adheres strictly to the schema contract, passes comprehensive
quality checks, and enforces strict train/serve isolation to prevent data leakage.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

from fraud_scoring.features import (
    IDENTITY_COLUMN,
    SERVING_FEATURES,
    TARGET_COLUMN,
    TRAINING_FEATURES,
)

Check = pa.Check
Column = pa.Column
DataFrameSchema = pa.DataFrameSchema

# =====================================================================
# 1. Feature Contracts & Column Definitions
# =====================================================================

TARGET: str = TARGET_COLUMN
# Backwards-compatible names for the Stage 2 validation interface. The values
# originate in features.py, which is the shared feature contract.
TRAIN_FEATURES: tuple[str, ...] = TRAINING_FEATURES

# Complete schema columns expected in raw CSV inputs
RAW_TRAIN_COLUMNS: List[str] = [
    IDENTITY_COLUMN,
    "amount",
    "merchant_category",
    "hour_of_day",
    "device_risk",
    TARGET,
]

RAW_SERVE_COLUMNS: List[str] = [
    IDENTITY_COLUMN,
    "amount",
    "merchant_category",
    "hour_of_day",
    "device_risk",
]

ALLOWED_MERCHANT_CATEGORIES: Set[str] = {
    "grocery",
    "electronics",
    "fashion",
    "travel",
    "gaming",
    "dining",
    "crypto",
    "utilities",
}

# Known post-authorization / post-transaction fields that must NEVER leak into features
FORBIDDEN_LEAKAGE_COLUMNS: Set[str] = {
    TARGET,
    "fraud_label",
    "chargeback_amount",
    "chargeback_date",
    "dispute_status",
    "authorization_result",
    "settlement_status",
    "post_auth_risk_score",
}

# Prefixes and suffixes reserved for columns derived from the label. They are
# blocked even when a new target-derived column was not added to the explicit
# denylist above.
TARGET_DERIVED_PREFIXES: tuple[str, ...] = ("target_", "is_fraud_", "fraud_label_")
TARGET_DERIVED_SUFFIXES: tuple[str, ...] = ("_target", "_is_fraud", "_fraud_label")


# =====================================================================
# 2. Custom Exceptions
# =====================================================================


class DataValidationError(Exception):
    """Raised when dataset fails schema, completeness, or range checks."""

    def __init__(self, message: str, errors: Optional[List[str]] = None):
        super().__init__(message)
        self.errors = errors or []


class DataLeakageError(Exception):
    """Raised when target label or post-auth fields leak into feature sets."""

    def __init__(self, message: str, leaked_columns: Optional[List[str]] = None):
        super().__init__(message)
        self.leaked_columns = leaked_columns or []


# =====================================================================
# 3. Validation Result
# =====================================================================


@dataclass
class ValidationResult:
    is_valid: bool
    total_records: int
    errors: List[str] = field(default_factory=list)
    quality_summary: Dict[str, Any] = field(default_factory=dict)

    def report(self) -> str:
        if self.is_valid:
            return f"Validation PASSED: {self.total_records} records checked. No errors detected."
        lines = [
            f"Validation FAILED: {self.total_records} records checked. Found {len(self.errors)} issues:",
        ]
        for err in self.errors:
            lines.append(f"  - {err}")
        return "\n".join(lines)


# =====================================================================
# 4. Pandera Schemas
# =====================================================================


def get_transaction_schema(is_training: bool = True) -> DataFrameSchema:
    """Build Pandera schema for transaction data validation."""
    columns = {
        IDENTITY_COLUMN: Column(
            pa.String,
            nullable=False,
            unique=True,
            title="Transaction Identity",
            description="Unique identity string for each transaction. Excluded from model features.",
        ),
        "amount": Column(
            pa.Float,
            checks=[
                Check.greater_than(0.0, error="Amount must be strictly greater than 0"),
                Check(lambda s: np.isfinite(s).all(), error="Amount must be finite"),
            ],
            nullable=False,
            title="Transaction Amount",
        ),
        "merchant_category": Column(
            pa.String,
            checks=[
                Check.isin(
                    ALLOWED_MERCHANT_CATEGORIES,
                    error=f"Merchant category must be in {ALLOWED_MERCHANT_CATEGORIES}",
                ),
            ],
            nullable=False,
            title="Merchant Category",
        ),
        "hour_of_day": Column(
            pa.Int,
            checks=[
                Check.in_range(0, 23, error="Hour of day must be an integer between 0 and 23"),
            ],
            nullable=False,
            title="Hour of Transaction",
        ),
        "device_risk": Column(
            pa.Float,
            checks=[
                Check.in_range(0.0, 1.0, error="Device risk must be a float between 0.0 and 1.0"),
                Check(lambda s: np.isfinite(s).all(), error="Device risk must be finite"),
            ],
            nullable=False,
            title="Device Risk Score",
        ),
    }

    if is_training:
        columns[TARGET] = Column(
            pa.Int,
            checks=[
                Check.isin([0, 1], error=f"Target {TARGET} must be either 0 or 1"),
            ],
            nullable=False,
            title="Target Label (Training only)",
        )

    return DataFrameSchema(
        columns=columns,
        strict=True,  # Disallows unexpected columns
        ordered=False,
    )


# =====================================================================
# 5. Leakage Guardrails
# =====================================================================


def check_leakage(feature_columns: Iterable[str], is_serving: bool = False) -> List[str]:
    """Inspect feature list and detect any target or post-auth leakage.

    Args:
        feature_columns: Iterable of feature names intended for model ingestion or serving.
        is_serving: Whether the feature list represents serving payload features.

    Returns:
        List of identified leakage violation descriptions.
    """
    col_set = set(feature_columns)
    violations: List[str] = []

    # Check 1: Target label leakage
    if TARGET in col_set:
        context = "serving feature payload" if is_serving else "model feature matrix X"
        violations.append(
            f"Target column '{TARGET}' detected in {context} (critical target leakage)."
        )

    # Check 2: Identity column leakage
    if IDENTITY_COLUMN in col_set:
        violations.append(
            f"Identity column '{IDENTITY_COLUMN}' detected in model features. ID must never be an input feature."
        )

    # Check 3: Post-auth / forbidden fields leakage
    forbidden_present = col_set.intersection(FORBIDDEN_LEAKAGE_COLUMNS - {TARGET})
    if forbidden_present:
        violations.append(
            f"Post-authorization columns detected: {sorted(forbidden_present)}. Only pre-auth features allowed."
        )

    # Check 4: Features directly derived from the training label.
    target_derived = sorted(
        column
        for column in col_set
        if column.startswith(TARGET_DERIVED_PREFIXES) or column.endswith(TARGET_DERIVED_SUFFIXES)
    )
    if target_derived:
        violations.append(
            f"Target-derived columns detected: {target_derived}. Labels may never be transformed into features."
        )

    return violations


def assert_no_leakage(feature_columns: Iterable[str], is_serving: bool = False) -> None:
    """Raise DataLeakageError if any leakage is detected in feature columns."""
    violations = check_leakage(feature_columns, is_serving=is_serving)
    if violations:
        raise DataLeakageError(
            f"Data leakage check failed with {len(violations)} violation(s):\n"
            + "\n".join(f"  - {v}" for v in violations),
            leaked_columns=violations,
        )


def assert_contract_invariants() -> None:
    """Verify built-in contract invariants on module load or test time."""
    # Invariant 1: TARGET must never be in SERVING_FEATURES
    if TARGET in SERVING_FEATURES:
        raise DataLeakageError(f"CRITICAL: {TARGET} is present in SERVING_FEATURES constant!")

    # Invariant 2: TARGET must never be in TRAIN_FEATURES
    if TARGET in TRAIN_FEATURES:
        raise DataLeakageError(f"CRITICAL: {TARGET} is present in TRAIN_FEATURES constant!")

    # Invariant 3: IDENTITY_COLUMN must not be in TRAIN_FEATURES
    if IDENTITY_COLUMN in TRAIN_FEATURES:
        raise DataLeakageError(
            f"CRITICAL: {IDENTITY_COLUMN} is present in TRAIN_FEATURES constant!"
        )

    # Invariant 4: SERVING_FEATURES must match TRAIN_FEATURES
    if sorted(SERVING_FEATURES) != sorted(TRAIN_FEATURES):
        raise ValueError(
            f"Contract mismatch: SERVING_FEATURES {SERVING_FEATURES} != TRAIN_FEATURES {TRAIN_FEATURES}"
        )


# Run invariants check immediately on import
assert_contract_invariants()


# =====================================================================
# 6. Data Quality & Comprehensive Validation
# =====================================================================


def validate_raw_transactions(
    df: pd.DataFrame,
    is_training: bool = True,
) -> ValidationResult:
    """Perform end-to-end schema, data quality, and leakage validation on raw transactions.

    Args:
        df: Input DataFrame to validate.
        is_training: If True, validates training schema (requires is_fraud). If False, validates serve schema.

    Returns:
        ValidationResult object containing boolean pass/fail status and detailed error messages.
    """
    errors: List[str] = []
    total_records = len(df)

    if total_records == 0:
        errors.append("Dataset is empty (0 records).")
        return ValidationResult(is_valid=False, total_records=0, errors=errors)

    # 1. Check for expected columns
    expected_cols = set(RAW_TRAIN_COLUMNS if is_training else RAW_SERVE_COLUMNS)
    actual_cols = set(df.columns)

    missing_cols = expected_cols - actual_cols
    if missing_cols:
        errors.append(f"Missing required columns: {sorted(missing_cols)}")

    unexpected_cols = actual_cols - expected_cols
    if unexpected_cols:
        errors.append(
            f"Unexpected columns detected (strict schema violation): {sorted(unexpected_cols)}"
        )

    # 2. Check for leakage if serving
    if not is_training and TARGET in df.columns:
        errors.append(f"Target column '{TARGET}' present in serving data payload (leakage).")

    # 3. Duplicate transaction_id check
    if IDENTITY_COLUMN in df.columns:
        dup_count = df[IDENTITY_COLUMN].duplicated().sum()
        if dup_count > 0:
            dup_examples = df[IDENTITY_COLUMN][df[IDENTITY_COLUMN].duplicated()].head(3).tolist()
            errors.append(
                f"Found {dup_count} duplicate '{IDENTITY_COLUMN}' values (e.g., {dup_examples})."
            )

    # 4. Missing value checks across expected columns
    for col in expected_cols.intersection(actual_cols):
        null_count = int(df[col].isna().sum())
        if null_count > 0:
            errors.append(f"Column '{col}' contains {null_count} missing (NaN/null) values.")

    # 5. Invalid amount values (<= 0 or non-numeric)
    if "amount" in df.columns and pd.api.types.is_numeric_dtype(df["amount"]):
        invalid_amounts = int((df["amount"] <= 0).sum())
        if invalid_amounts > 0:
            errors.append(f"Column 'amount' contains {invalid_amounts} non-positive (<= 0) values.")

    # 6. Invalid hour_of_day values
    if "hour_of_day" in df.columns and pd.api.types.is_numeric_dtype(df["hour_of_day"]):
        invalid_hours = int(((df["hour_of_day"] < 0) | (df["hour_of_day"] > 23)).sum())
        if invalid_hours > 0:
            errors.append(f"Column 'hour_of_day' contains {invalid_hours} values outside [0, 23].")

    # 7. Invalid device_risk values
    if "device_risk" in df.columns and pd.api.types.is_numeric_dtype(df["device_risk"]):
        invalid_risk = int(((df["device_risk"] < 0.0) | (df["device_risk"] > 1.0)).sum())
        if invalid_risk > 0:
            errors.append(
                f"Column 'device_risk' contains {invalid_risk} values outside [0.0, 1.0]."
            )

    # 8. Invalid merchant categories
    if "merchant_category" in df.columns:
        invalid_cats = set(df["merchant_category"].dropna().unique()) - ALLOWED_MERCHANT_CATEGORIES
        if invalid_cats:
            errors.append(
                f"Column 'merchant_category' contains unauthorized categories: {sorted(invalid_cats)}"
            )

    # 9. Target validity (training only)
    if is_training and TARGET in df.columns and pd.api.types.is_numeric_dtype(df[TARGET]):
        invalid_targets = set(df[TARGET].dropna().unique()) - {0, 1}
        if invalid_targets:
            errors.append(
                f"Column '{TARGET}' contains invalid target values: {sorted(invalid_targets)} (expected {{0, 1}})."
            )

    # 10. Pandera Schema validation
    try:
        schema = get_transaction_schema(is_training=is_training)
        schema.validate(df, lazy=True)
    except SchemaErrors as se:
        for err in se.failure_cases["check"].unique():
            err_msg = f"Pandera schema check failed: {err}"
            if err_msg not in errors:
                errors.append(err_msg)
    except Exception as e:
        err_msg = f"Pandera validation error: {e}"
        if err_msg not in errors:
            errors.append(err_msg)

    # Compile quality summary
    quality_summary = {
        "total_records": total_records,
        "is_training": is_training,
        "fraud_rate": float(df[TARGET].mean()) if is_training and TARGET in df.columns else None,
        "amount_mean": float(df["amount"].mean())
        if "amount" in df.columns and pd.api.types.is_numeric_dtype(df["amount"])
        else None,
        "error_count": len(errors),
    }

    is_valid = len(errors) == 0
    return ValidationResult(
        is_valid=is_valid,
        total_records=total_records,
        errors=errors,
        quality_summary=quality_summary,
    )


def validate_and_enforce(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """Validate DataFrame and return it if valid; raises DataValidationError if invalid."""
    result = validate_raw_transactions(df, is_training=is_training)
    if not result.is_valid:
        raise DataValidationError(result.report(), errors=result.errors)
    return df


# =====================================================================
# 7. CLI Entrypoint for DVC / CI Pipeline Stage
# =====================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate transaction data quality, schema, and leakage"
    )
    parser.add_argument(
        "--data-path", default="data/raw/transactions.csv", help="Path to CSV dataset"
    )
    parser.add_argument(
        "--mode", choices=["train", "serve"], default="train", help="Validation mode"
    )
    args = parser.parse_args()

    data_file = Path(args.data_path)
    if not data_file.exists():
        print(f"ERROR: Dataset not found at {data_file}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading dataset: {data_file} (Mode: {args.mode})...")
    df = pd.read_csv(data_file)

    is_training = args.mode == "train"
    result = validate_raw_transactions(df, is_training=is_training)

    if result.is_valid:
        print(result.report())
        if result.quality_summary.get("fraud_rate") is not None:
            print(f"Fraud Rate: {result.quality_summary['fraud_rate']:.2%}")
        sys.exit(0)
    else:
        print(result.report(), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
