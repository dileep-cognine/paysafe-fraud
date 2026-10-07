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
        │
        ▼
MLflow run (dataset fingerprint, parameters, metrics, feature contract,
            confusion matrix, signature, input example)
        │
        ▼
registered model: paysafe-fraud-detector
        │
        ▼
approved alias: @champion
        │
        ▼
FastAPI startup resolves and pins the approved version
        │
        ▼
POST /score → risk_score + model_version
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

The DVC stages do not hardcode a profile. Without an explicit CLI profile, the
shared loader selects `configs/<APP_ENV>.yaml`; CI sets `APP_ENV=ci`, while a
normal local run defaults to `dev`. `CONFIG_PATH` intentionally overrides that
selection when a reviewer needs a specific profile.

`prepare` fails on schema, quality, or leakage violations. `evaluate` writes
the measured metrics and exits nonzero when its configured quality gate fails.
The independent MLflow command records a complete real experiment for a
validated dataset; promotion subsequently rechecks that recorded gate before
the `champion` alias can move.

The DVC pointer and lock file identify the raw dataset and reproducible stage
dependencies. MLflow records the dataset SHA-256, DVC digest when the tracked
raw CSV is supplied, run ID, logged-model URI, evaluation artifacts, and model
signature. Registry promotion records the source run; serving returns the exact
pinned registry version in every `/score` response.

The current DVC remote is local. To migrate later, configure an approved remote
without claiming it is already active:

```bash
dvc remote add -d production s3://<bucket>/<prefix>
# or
dvc remote add -d production gs://<bucket>/<prefix>
dvc push
```

Authenticate S3/GCS through the deployment environment or credential manager;
do not commit credentials to DVC configuration or `.env.example`.
