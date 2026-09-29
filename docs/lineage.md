# Data and model lineage

```text
Synthetic transaction CSV
        ↓
DVC validate_data stage (dvc.yaml)
        ↓
shared schema, quality, leakage, and feature contracts
        ↓
Git commit recorded when an MLflow training run is created
        ↓
MLflow experiment run
  ├─ parameters and preprocessing description
  ├─ evaluation metrics and quality-gate tag
  ├─ dataset path, byte size, row count, and SHA-256 tag
  ├─ metrics artifact
  └─ fitted model artifact, signature, and input example
        ↓
explicitly registered model version after passing promotion checks
        ↓
configured champion alias
        ↓
Docker image built from the matching repository revision
        ↓
FastAPI process pins the champion version at startup
        ↓
POST /score response includes that pinned model_version
```

The repository intentionally does not place a fabricated DVC hash, Git hash,
MLflow run ID, model version, or Docker image digest in this document. Inspect
the actual MLflow run and model registry during a walkthrough. Training records
the raw-data SHA-256 and Git commit only when they exist; promotion records the
training run on the registered version.

The local synthetic CSV and local SQLite MLflow store are development aids. A
production container requires a network-reachable MLflow tracking/registry
service with accessible artifacts; it does not rely on host Windows file paths.
