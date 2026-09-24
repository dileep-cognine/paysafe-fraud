# PaySafe Fraud Scoring - Dataset & Data Validation Documentation

> **Notice**: The transaction data stored in `data/raw/transactions.csv` is **realistic synthetic demonstration data** generated deterministically via `src/fraud_scoring/generate_data.py`. No actual customer or PII data is stored.

---

## 1. Dataset Schema & Feature Contracts

The dataset models card and digital payment transactions captured at authorization time:

| Column Name | Type | Train | Serve | Description | Validation Rule |
| :--- | :--- | :---: | :---: | :--- | :--- |
| `transaction_id` | string | Yes | Yes | Unique transaction identifier | Non-null, unique across dataset, excluded from model feature sets |
| `amount` | float | Yes | Yes | Transaction value | Strictly positive (`amount > 0.0`), finite float |
| `merchant_category` | string | Yes | Yes | Categorical merchant category | Must be in allowed controlled vocabulary |
| `hour_of_day` | int | Yes | Yes | Local hour of transaction initiation | Integer in range `[0, 23]` |
| `device_risk` | float | Yes | Yes | Device fingerprint risk score | Float in range `[0.0, 1.0]` |
| `is_fraud` | int | **Yes** | **NO** | Binary fraud indicator (ground truth) | Value in `{0, 1}`. **Training only; strictly forbidden at serving** |

### Controlled Merchant Categories
The `merchant_category` field accepts only the following controlled categories:
- `grocery`
- `electronics`
- `fashion`
- `travel`
- `gaming`
- `dining`
- `crypto`
- `utilities`

Any unfamiliar category triggers a schema validation error to prevent silent distribution drifts.

---

## 2. Training-Only vs Serving Features

To prevent training-serving skew and data leakage:

```text
TRAIN_FEATURES = ["amount", "merchant_category", "hour_of_day", "device_risk"]
TARGET         = "is_fraud"
IDENTITY       = "transaction_id"

SERVING_FEATURES = ["amount", "merchant_category", "hour_of_day", "device_risk"]
```

- **`transaction_id`** is retained for logging, auditing, and observability, but is explicitly stripped from model input matrices $X$.
- **`is_fraud`** is strictly prohibited from `SERVING_FEATURES`. Its presence in serving requests immediately raises a `DataLeakageError`.

---

## 3. Shared Feature Builder and Train/Serve Consistency

`src/fraud_scoring/features.py` owns the ordered model contract and is the only place that builds estimator input:

```text
transaction_id  -> identity and audit context only
amount          -> float64 model feature
merchant_category -> string model feature
hour_of_day     -> int64 model feature
device_risk     -> float64 model feature
is_fraud        -> int64 training target only
```

`build_training_features()` returns the model matrix and target. `build_serving_features()` returns the same model matrix from a label-free authorization request. Both call the same internal transformation, enforce the same input types, and return the same fixed column order. The builder has no fitted state or learned encoding yet; a later training stage will fit preprocessing once and reuse it for inference.

`tests/test_train_serve_consistency.py` supplies the same raw transactions to both paths and asserts identical columns, order, values, and dtypes. It also asserts that neither the identity nor target reaches a model matrix.

---

## 4. Data Quality Rules & Gates

The validation suite (`src/fraud_scoring/data_validation.py`) enforces:

1. **Completeness**: Zero missing (null/NaN) values across all required contract columns.
2. **Uniqueness**: `transaction_id` must be 100% unique (no duplicates).
3. **Value Domain**:
   - `amount > 0.0`
   - `0 <= hour_of_day <= 23`
   - `0.0 <= device_risk <= 1.0`
4. **Vocabulary Conformance**: `merchant_category` must be an exact match to allowed values.
5. **Strict Schema**: Unexpected/unapproved columns are rejected to prevent unversioned or leaky features from bypassing gates.
6. **Binary Label Integrity**: Training labels must be strictly `0` or `1`.

---

## 5. Leakage Prevention Architecture

Data leakage is defended against at multiple layers:

1. **Contract Invariants**: Checked at Python module import time:
   - Asserts `is_fraud` is not in `SERVING_FEATURES`.
   - Asserts `is_fraud` is not in `TRAIN_FEATURES`.
   - Asserts `transaction_id` is not in `TRAIN_FEATURES`.
2. **Schema Ingestion Gate**: `validate_raw_transactions(df, is_training=False)` rejects any serving payload containing `is_fraud`.
3. **Forbidden Field Denylist**: Explicitly blocks post-authorization columns (e.g., `chargeback_amount`, `dispute_status`, `settlement_status`).
4. **Target-Derived Field Guard**: Names such as `target_rolling_fraud_rate` and `is_fraud_encoded` are rejected; the label must never be transformed into a feature.
5. **Guard Function**: `assert_no_leakage(columns)` is called before model matrix construction to block rogue features.

The schema validates values as supplied. It never coerces strings to numbers or repairs malformed values: an input contract violation is reported and the validation command exits non-zero.

---

## 6. How to Run Validation & Tests

### Run Data Quality Validation on Raw Data
```bash
# Validate training dataset
python -m fraud_scoring.data_validation --data-path data/raw/transactions.csv --mode train

# The raw sample is labelled training data. Supplying it with --mode serve is
# expected to fail because the label is forbidden from serving payloads.
python -m fraud_scoring.data_validation --data-path data/raw/transactions.csv --mode serve
```

### Regenerate Synthetic Transactions
```bash
python -m fraud_scoring.generate_data --records 5000 --fraud-ratio 0.04
```

### Run Automated Tests
```bash
pytest tests/test_data_validation.py
pytest tests/test_features.py tests/test_train_serve_consistency.py
```
