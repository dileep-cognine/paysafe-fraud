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
```

Git tracks `dvc.yaml`, `dvc.lock`, `data/raw/transactions.csv.dvc`, source,
configuration, and tests. The raw CSV itself and the derived outputs are DVC
managed and ignored by Git. MLflow records experiment evidence and registry
versions but does not replace the DVC data lineage.

Use these commands from the repository root:

```bash
# Restore the tracked raw data from the configured remote when needed.
dvc pull

# Reproduce only changed stages and inspect the dependency graph/state.
dvc repro
dvc dag
dvc status

# Share updated DVC data/cache with the configured remote after review.
dvc push
```

`prepare` fails on schema, quality, or leakage violations. `evaluate` writes
the measured metrics and exits nonzero when its configured quality gate fails.
The independent MLflow command records a complete real experiment for a
validated dataset; promotion subsequently rechecks that recorded gate before
the `champion` alias can move.
