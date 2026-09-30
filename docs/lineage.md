# Data and model lineage

The reproducible data/model path is declared in `dvc.yaml`:

```text
data/raw/transactions.csv
        │
        │ DVC-tracked dataset
        ▼
prepare
  validate schema, quality, leakage, and deterministic merchant normalization
        │
        ▼
data/processed/validated.csv
        │
        ▼
train
  fit the configured logistic-regression candidate
        │
        ▼
artifacts/model/candidate.joblib
        │
        ▼
evaluate
  calculate metrics and apply configured quality thresholds
        │
        ▼
artifacts/evaluation/metrics.json