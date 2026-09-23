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

## 3. Data Quality Rules & Gates

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

## 4. Leakage Prevention Architecture

Data leakage is defended against at multiple layers:

1. **Contract Invariants**: Checked at Python module import time:
   - Asserts `is_fraud` is not in `SERVING_FEATURES`.
   - Asserts `is_fraud` is not in `TRAIN_FEATURES`.
   - Asserts `transaction_id` is not in `TRAIN_FEATURES`.
2. **Schema Ingestion Gate**: `validate_raw_transactions(df, is_training=False)` rejects any serving payload containing `is_fraud`.
3. **Forbidden Field Denylist**: Explicitly blocks post-authorization columns (e.g., `chargeback_amount`, `dispute_status`, `settlement_status`).
4. **Guard Function**: `assert_no_leakage(columns)` is called before model matrix construction to block rogue features.

---

## 5. How to Run Validation & Tests

### Run Data Quality Validation on Raw Data
```bash
# Validate training dataset
python -m fraud_scoring.data_validation --data-path data/raw/transactions.csv --mode train

# Validate serving payload format
python -m fraud_scoring.data_validation --data-path data/raw/transactions.csv --mode serve
```

### Regenerate Synthetic Transactions
```bash
python -m fraud_scoring.generate_data --records 5000 --fraud-ratio 0.04
```

### Run Automated Tests
```bash
pytest tests/test_data_validation.py
```
